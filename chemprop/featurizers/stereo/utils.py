import numpy as np
from rdkit import Chem


def is_odd_permutation(i: int, j: int, k: int, m: int | None = None) -> int:
    r"""Check if a permutation is odd."""
    swaps = int(i > j) + int(i > k) + int(j > k)
    if m is not None:
        swaps += int(i > m) + int(j > m) + int(k > m)
    return bool(swaps % 2)


def assign_neighbor_ranking(mol: Chem.Mol, force: bool = False) -> None:
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

    >>> mol = Chem.MolFromSmiles("NC(O)=C(OC)OC")
    >>> assign_neighbor_ranking(mol)
    >>> for atom in mol.GetAtoms():
    ...     if len(atom.GetNeighbors()) > 1:
    ...         print(neighbor_ranking_string(atom))
    C1 C3:0 O2:1 N0:2
    C3 C1:0 O4:1 O6:1
    O4 C3:0 C5:1
    O6 C3:0 C7:1

    Assign neighbor ranks to molecules with a tetrahedral chiral center

    >>> for smiles in ["C[C@](O)(S)N", "C[C@@](S)(O)N"]:
    ...     print(f"\nMolecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print(neighbor_ranking_string(mol.GetAtomWithIdx(1)))
    ...     print("Flip tag:", mol.GetAtomWithIdx(1).GetBoolProp("flipChiralTag"))
    <BLANKLINE>
    Molecule: C[C@](O)(S)N
    C1 S3:0 O2:1 N4:2 C0:3
    Flip tag: False
    <BLANKLINE>
    Molecule: C[C@@](S)(O)N
    C1 S2:0 O3:1 N4:2 C0:3
    Flip tag: True

    Assign neighbor ranks to molecules with a stereogenic double bond:

    >>> for smiles in [r"CC(/Cl)=C(N)/C", r"CC(/Cl)=C(\N)C"]:
    ...     print(f"\nMolecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     for index in [1, 3]:
    ...         print(neighbor_ranking_string(mol.GetAtomWithIdx(index)))
    ...     for set_from_directions in [False, True]:
    ...         if set_from_directions:
    ...             Chem.SetBondStereoFromDirections(mol)
    ...             assign_neighbor_ranking(mol, force=True)
    ...         bond = mol.GetBondWithIdx(2)
    ...         print("Originally:", bond.GetStereo().name, *bond.GetStereoAtoms())
    ...         print("Flip stereo:", bond.GetBoolProp("flipStereo"))
    <BLANKLINE>
    Molecule: CC(/Cl)=C(N)/C
    C1 C3:0 Cl2:1 C0:2
    C3 C1:0 N4:1 C5:2
    Originally: STEREOE 2 4
    Flip stereo: False
    Originally: STEREOCIS 2 5
    Flip stereo: True
    <BLANKLINE>
    Molecule: CC(/Cl)=C(\N)C
    C1 C3:0 Cl2:1 C0:2
    C3 C1:0 N4:1 C5:2
    Originally: STEREOE 2 4
    Flip stereo: False
    Originally: STEREOTRANS 2 4
    Flip stereo: False

    """
    if not force and mol.HasProp("hasNeighborRanks") and mol.GetBoolProp("hasNeighborRanks"):
        return
    atom_ranks = Chem.CanonicalRankAtoms(
        mol, breakTies=False, includeChirality=False, includeAtomMaps=False
    )
    atom_priorities = -np.fromiter(atom_ranks, dtype=int)
    sorted_neighbors = []
    for atom in mol.GetAtoms():
        neighbors = np.fromiter((atom.GetIdx() for atom in atom.GetNeighbors()), dtype=int)
        neighbor_priorities = atom_priorities[neighbors]
        ranks = np.searchsorted(np.sort(neighbor_priorities), neighbor_priorities)
        sorted_neighbors.append(dict(zip(neighbors, ranks)))

        if atom.GetChiralTag() in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }:
            atom.SetBoolProp("flipChiralTag", is_odd_permutation(*ranks))

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endRankFromBegin", int(sorted_neighbors[begin][end]))
        bond.SetIntProp("beginRankFromEnd", int(sorted_neighbors[end][begin]))

        if bond.GetStereo() in {
            Chem.BondStereo.STEREOCIS,
            Chem.BondStereo.STEREOTRANS,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOE,
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


def neighbor_ranking_string(atom: Chem.Atom) -> None:
    """Return a string representation of the neighbor ranking for a given atom.

    Parameters
    ----------
    atom : Chem.Atom
        The RDKit atom for which the neighbor ranking string is generated.

    Returns
    -------
    str
        A string that lists the atom itself followed by its neighbors and their ranks.
        The format is "AtomSymbolAtomIndex NeighborSymbolNeighborIndex:Rank ...",
        with neighbors sorted by their rank.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> assign_neighbor_ranking(mol)
    >>> atom = mol.GetAtomWithIdx(1)
    >>> neighbor_ranking_string(atom)
    'C1 O2:0 C0:1'

    """
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
        output.append(f"{neighbor.GetSymbol()}{neighbor.GetIdx()}:{rank}")
    return " ".join(output)


def get_canonical_chiral_tag(atom: Chem.Atom) -> Chem.ChiralType:
    """Return the canonical chiral tag of an atom.

    Parameters
    ----------
    atom : Chem.Atom
        The atom whose chiral tag is to be returned.

    Returns
    -------
    Chem.ChiralType
        The chiral tag of the atom.

    Example
    -------
    >>> for smiles in ["C[C@](O)(S)N", "C[C@](S)(O)N"]:
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     atom = mol.GetAtomWithIdx(1)
    ...     print(neighbor_ranking_string(atom), get_canonical_chiral_tag(atom).name)
    C1 S3:0 O2:1 N4:2 C0:3 CHI_TETRAHEDRAL_CCW
    C1 S2:0 O3:1 N4:2 C0:3 CHI_TETRAHEDRAL_CW

    """
    chiral_tag = atom.GetChiralTag()
    if chiral_tag not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:
        return chiral_tag
    flip = atom.HasProp("flipChiralTag") and atom.GetBoolProp("flipChiralTag")
    if (chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW) == flip:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    return Chem.ChiralType.CHI_TETRAHEDRAL_CW


def get_canonical_stereo(bond: Chem.Bond) -> Chem.BondStereo:
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
    Chem.BondStereo
        The stereochemistry flag of the bond.

    Example
    -------
    >>> for smiles in [r"CC(/Cl)=C(N)/C", r"CC(/Cl)=C(\N)F"]:
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     atoms = map(mol.GetAtomWithIdx, [1, 3])
    ...     bond = mol.GetBondBetweenAtoms(1, 3)
    ...     assign_neighbor_ranking(mol)
    ...     print(*map(neighbor_ranking_string, atoms), get_canonical_stereo(bond).name)
    C1 C3:0 Cl2:1 C0:2 C3 C1:0 N4:1 C5:2 STEREOTRANS
    C1 C3:0 Cl2:1 C0:2 C3 C1:0 F5:1 N4:2 STEREOCIS

    """
    stereo = bond.GetStereo()
    if stereo not in {
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOE,
    }:
        return stereo
    flip = bond.HasProp("flipStereo") and bond.GetBoolProp("flipStereo")
    if (stereo in {Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ}) == flip:
        return Chem.BondStereo.STEREOTRANS
    return Chem.BondStereo.STEREOCIS
