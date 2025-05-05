import numpy as np
from rdkit.Chem.rdchem import Atom, Bond, BondStereo, ChiralType, Mol
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms


def is_odd_permutation(i: int, j: int, k: int, m: int | None = None) -> int:
    r"""Check if a permutation is odd."""
    swaps = int(i > j) + int(i > k) + int(j > k)
    if m is not None:
        swaps += int(i > m) + int(j > m) + int(k > m)
    return bool(swaps % 2)


def assign_neighbor_ranking(mol: Mol, force: bool = False) -> None:
    r"""Assign canonical neighbor ranks and indicate whether to flip stereochemical tags.

    The neighbors of each atom in the molecule are sorted in descending order of their canonical
    ranks (as assigned by :func:`rdkit.Chem.CanonicalRankAtoms`). The resulting relative ranking
    indicates the neighbor's position in this sorted list, with lower values corresponding to
    higher canonical ranks. Hydrogens are placed at the end of the list due to their low atomic
    number and connectivity.

    The relative ranking is stored as integer properties on each bond in the molecule:

    - `endRankFromBegin`: the relative rank of the bond's end atom among the neighbors of the
      begin atom.
    - `beginRankFromEnd`: the relative rank of the bond's begin atom among the neighbors of the
      end atom.

    For example, neighbors with canonical ranks `[3, 1, 4]` will have relative ranks `[1, 2, 0]`.
    Ties result in the same relative rank. For example, if the canonical ranks are `[3, 1, 1]`,
    the relative ranks will be `[2, 0, 0]`.

    A boolean property `flipChiralTag` is added to each tetrahedral chiral center (i.e., an atom
    with chiral tag `CHI_TETRAHEDRAL_CW` or `CHI_TETRAHEDRAL_CCW`). Another boolean property
    `flipStereo` is added to each stereogenic double bond (i.e., one with stereo tag `STEREOCIS`,
    `STEREOTRANS`, `STEREOZ`, or `STEREOE`). These properties indicate whether the tag should be
    flipped to match the canonical neighbor ranking.

    The molecule is also tagged with a boolean property ``hasNeighborRanks`` set to True to
    indicate that relative neighbor ranking has been computed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule whose bonds and stereocenters will be annotated.
    force : bool, default=False
        Whether to recompute rankings and overwrite any existing annotations, even if
        the molecule already has a ``hasNeighborRanks`` property set to True.

    Examples
    --------
    Assign neighbor ranks to a molecule

    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("NC(O)=C(OC)OC")
    >>> assign_neighbor_ranking(mol)
    >>> print(describe_neighbor_ranking(mol))
    C1 C3:0 O2:1 N0:2
    C3 C1:0 O4:1 O6:1
    O4 C3:0 C5:1
    O6 C3:0 C7:1

    Assign neighbor ranks to molecules with a tetrahedral chiral center

    >>> for smiles in ["C[C@](O)(S)N", "C[C@@](S)(O)N"]:
    ...     print(f"Molecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print(describe_neighbor_ranking(mol))
    Molecule: C[C@](O)(S)N
    C1 S3:0 O2:1 N4:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    Molecule: C[C@@](S)(O)N
    C1 S2:0 O3:1 N4:2 C0:3 (CHI_TETRAHEDRAL_CCW)

    Assign neighbor ranks to molecules with a stereogenic double bond:

    >>> for smiles in [r"CC(/Cl)=C(N)/C", r"CC(/Cl)=C(\N)C"]:
    ...     print(f"Molecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     for set_from_directions in [False, True]:
    ...         if set_from_directions:
    ...             Chem.SetBondStereoFromDirections(mol)
    ...             assign_neighbor_ranking(mol, force=True)
    ...         print(describe_neighbor_ranking(mol))
    Molecule: CC(/Cl)=C(N)/C
    Molecule does not have neighbor ranks.
    C1 C3:0 Cl2:1 C0:2
    C3 C1:0 N4:1 C5:2
    C1-C3 STEREOTRANS
    Molecule: CC(/Cl)=C(\N)C
    Molecule does not have neighbor ranks.
    C1 C3:0 Cl2:1 C0:2
    C3 C1:0 N4:1 C5:2
    C1-C3 STEREOTRANS

    """
    if not force and mol.HasProp("hasNeighborRanks") and mol.GetBoolProp("hasNeighborRanks"):
        return
    atom_ranks = CanonicalRankAtoms(
        mol, breakTies=False, includeChirality=False, includeAtomMaps=False
    )
    atom_priorities = -np.fromiter(atom_ranks, dtype=int)
    sorted_neighbors = []
    for atom in mol.GetAtoms():
        neighbors = np.fromiter((atom.GetIdx() for atom in atom.GetNeighbors()), dtype=int)
        neighbor_priorities = atom_priorities[neighbors]
        ranks = np.searchsorted(np.sort(neighbor_priorities), neighbor_priorities)
        sorted_neighbors.append(dict(zip(neighbors, ranks)))

        if atom.GetChiralTag() in {ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW}:
            atom.SetBoolProp("flipChiralTag", is_odd_permutation(*ranks))

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endRankFromBegin", int(sorted_neighbors[begin][end]))
        bond.SetIntProp("beginRankFromEnd", int(sorted_neighbors[end][begin]))

        if bond.GetStereo() in {
            BondStereo.STEREOCIS,
            BondStereo.STEREOTRANS,
            BondStereo.STEREOZ,
            BondStereo.STEREOE,
        }:
            left, right = bond.GetStereoAtoms()
            left_rank_is_min = not any(
                v < sorted_neighbors[begin][left]
                for k, v in sorted_neighbors[begin].items()
                if k != end
            )
            right_rank_is_min = not any(
                v < sorted_neighbors[end][right]
                for k, v in sorted_neighbors[end].items()
                if k != begin
            )
            bond.SetBoolProp("flipStereo", left_rank_is_min != right_rank_is_min)

    mol.SetBoolProp("hasNeighborRanks", True)


def describe_neighbor_ranking(mol: Mol, include_leaves: bool = False) -> str:
    """Return a string representation of the neighbor ranking for all atoms in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule for which the neighbor ranking strings are generated.
    include_leaves : bool, optional
        Whether to include leaves (i.e., atoms with only one neighbor) in the neighbor ranking
        strings. The default is ``False``.

    Returns
    -------
    str
        A string that lists the neighbor ranking strings for all atoms in the molecule.

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")
    >>> print(describe_neighbor_ranking(mol))
    Molecule does not have neighbor ranks.
    >>> assign_neighbor_ranking(mol)
    >>> print(describe_neighbor_ranking(mol))
    C1 C3:0 O2:1 N6:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    C3 C1:0 C4:1
    C4 C3:0 O5:1
    C3-C4 STEREOTRANS
    >>> print(describe_neighbor_ranking(mol, include_leaves=True))
    C0 C1:0
    C1 C3:0 O2:1 N6:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    O2 C1:0
    C3 C1:0 C4:1
    C4 C3:0 O5:1
    O5 C4:0
    N6 C1:0
    C3-C4 STEREOTRANS

    """
    if not (mol.HasProp("hasNeighborRanks") and mol.GetBoolProp("hasNeighborRanks")):
        return "Molecule does not have neighbor ranks."

    def atom_str(atom: Atom) -> str:
        return f"{atom.GetSymbol()}{atom.GetIdx()}"

    lines = []
    for atom in mol.GetAtoms():
        bonds = atom.GetBonds()
        if len(bonds) > 1 or include_leaves:
            atom_idx = atom.GetIdx()
            neighbors = []
            for bond in atom.GetBonds():
                begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
                if begin.GetIdx() == atom_idx:
                    neighbor = end
                    rank = bond.GetIntProp("endRankFromBegin")
                else:
                    neighbor = begin
                    rank = bond.GetIntProp("beginRankFromEnd")
                neighbors.append((neighbor, rank))
            neighbors.sort(key=lambda x: x[1])
            output = [f"{atom.GetSymbol()}{atom_idx}"]
            for neighbor, rank in neighbors:
                output.append(f"{atom_str(neighbor)}:{rank}")
            tag = get_canonical_chiral_tag(atom)
            if tag != ChiralType.CHI_UNSPECIFIED:
                output.append(f"({tag.name})")
            lines.append(" ".join(output))
    for bond in mol.GetBonds():
        tag = get_canonical_stereo(bond)
        if tag != BondStereo.STEREONONE:
            begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
            lines.append(f"{atom_str(begin)}-{atom_str(end)} {tag.name}")
    return "\n".join(lines)


def get_canonical_chiral_tag(atom: Atom) -> ChiralType:
    """Return the canonical chiral tag of an atom.

    Parameters
    ----------
    atom : Chem.Atom
        The atom whose chiral tag is to be returned.

    Returns
    -------
    ChiralType
        The chiral tag of the atom.

    Example
    -------
    >>> from rdkit import Chem
    >>> for smiles in ["C[C@](O)(S)N", "C[C@](S)(O)N", "C[C@@](S)(O)N"]:
    ...     print(f"Molecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print(describe_neighbor_ranking(mol))
    Molecule: C[C@](O)(S)N
    C1 S3:0 O2:1 N4:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    Molecule: C[C@](S)(O)N
    C1 S2:0 O3:1 N4:2 C0:3 (CHI_TETRAHEDRAL_CW)
    Molecule: C[C@@](S)(O)N
    C1 S2:0 O3:1 N4:2 C0:3 (CHI_TETRAHEDRAL_CCW)

    """
    chiral_tag = atom.GetChiralTag()
    if chiral_tag not in {ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW}:
        return chiral_tag
    flip = atom.HasProp("flipChiralTag") and atom.GetBoolProp("flipChiralTag")
    if (chiral_tag == ChiralType.CHI_TETRAHEDRAL_CW) == flip:
        return ChiralType.CHI_TETRAHEDRAL_CCW
    return ChiralType.CHI_TETRAHEDRAL_CW


def get_canonical_stereo(bond: Bond) -> BondStereo:
    r"""Return the canonical stereochemistry flag of a bond.

    .. note::
        This function never returns ``STEREOZ`` or ``STEREOE`` flags. Instead, it returns
        ``STEREOCIS`` or ``STEREOTRANS`` flags, respectively.

    Parameters
    ----------
    bond : Chem.Bond
        The bond whose stereochemistry flag is to be returned.

    Returns
    -------
    BondStereo
        The stereochemistry flag of the bond.

    Example
    -------
    >>> from rdkit import Chem
    >>> for smiles in [r"CC(/Cl)=C(N)/C", r"CC(/Cl)=C(\N)F"]:
    ...     print(f"Molecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print(describe_neighbor_ranking(mol))
    Molecule: CC(/Cl)=C(N)/C
    C1 C3:0 Cl2:1 C0:2
    C3 C1:0 N4:1 C5:2
    C1-C3 STEREOTRANS
    Molecule: CC(/Cl)=C(\N)F
    C1 C3:0 Cl2:1 C0:2
    C3 C1:0 F5:1 N4:2
    C1-C3 STEREOCIS

    """
    stereo = bond.GetStereo()
    if stereo not in {
        BondStereo.STEREOCIS,
        BondStereo.STEREOTRANS,
        BondStereo.STEREOZ,
        BondStereo.STEREOE,
    }:
        return stereo
    flip = bond.HasProp("flipStereo") and bond.GetBoolProp("flipStereo")
    if (stereo in {BondStereo.STEREOCIS, BondStereo.STEREOZ}) == flip:
        return BondStereo.STEREOTRANS
    return BondStereo.STEREOCIS
