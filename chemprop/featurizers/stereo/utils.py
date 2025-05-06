import numpy as np
from rdkit.Chem.rdchem import Atom, Bond, BondStereo, ChiralType, Mol
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms

BEGIN_RANK = "beginRank"
END_RANK = "endRank"
FLIP_CHIRAL_TAG = "flipChiralTag"
FLIP_STEREO = "flipStereo"
HAS_NEIGHBOR_RANKS = "hasNeighborRanks"


def is_odd_permutation(i: int, j: int, k: int, m: int | None = None) -> int:
    r"""Check if a permutation is odd."""
    swaps = int(i > j) + int(i > k) + int(j > k)
    if m is not None:
        swaps += int(i > m) + int(j > m) + int(k > m)
    return bool(swaps % 2)


def assign_neighbor_ranking(mol: Mol, all_atoms: bool = False, force: bool = False) -> None:
    f"""Assign canonical neighbor ranks and indicate whether to flip stereochemical tags.

    The neighbors of each atom in the molecule are sorted in descending order of their canonical
    ranks (as assigned by :func:`rdkit.Chem.CanonicalRankAtoms`). The resulting relative ranking
    indicates the neighbor's position in this sorted list, with lower values corresponding to
    higher canonical ranks. Hydrogens are placed at the end of the list due to their low atomic
    number and connectivity.
    
    For example, neighbors with canonical ranks `[3, 1, 4]` will have relative ranks `[1, 2, 0]`.
    Ties result in the same relative rank. For example, if the canonical ranks are `[3, 1, 1]`,
    the relative ranks will be `[2, 0, 0]`.

    An exception is made for the atoms engaged in a cis-trans double bond, which are always
    each other's first neighbors.

    By default, only chiral centers and atoms engaged in a cis-trans double bonds have their
    neighbors ranked. If ``all_atoms`` is ``True``, neighbors of all atoms are ranked.

    The relative ranking is stored as integer properties on bonds:

    - `{BEGIN_RANK}`: the relative rank of the bond's end atom among the neighbors of the
      begin atom.
    - `{END_RANK}`: the relative rank of the bond's begin atom among the neighbors of the
      end atom.

    A boolean property `{FLIP_CHIRAL_TAG}` is added to each tetrahedral chiral center (i.e., an atom
    with chiral tag `CHI_TETRAHEDRAL_CW` or `CHI_TETRAHEDRAL_CCW`). Another boolean property
    `{FLIP_STEREO}` is added to each stereogenic double bond (i.e., one with stereo tag `STEREOCIS`,
    `STEREOTRANS`, `STEREOZ`, or `STEREOE`). These properties indicate whether the tag should be
    flipped to match the canonical neighbor ranking.

    The molecule is also tagged with a boolean property ``{HAS_NEIGHBOR_RANKS}`` set to True to
    indicate that relative neighbor ranking has been computed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule whose bonds and stereocenters will be annotated.
    force : bool, default=False
        Whether to recompute rankings and overwrite any existing annotations, even if
        the molecule already has a ``{HAS_NEIGHBOR_RANKS}`` property set to True.

    Examples
    --------
    Assign neighbor ranks to a molecule

    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("NC(O)=C(OC)OC")
    >>> assign_neighbor_ranking(mol)
    >>> print(describe_neighbor_ranking(mol))
    <BLANKLINE>

    Assign neighbor ranks to molecules with a tetrahedral chiral center

    >>> for smiles in ["C[C@](O)(S)N", "C[C@@](S)(O)N"]:
    ...     print(f"Molecule: {{smiles}}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print(describe_neighbor_ranking(mol))
    Molecule: C[C@](O)(S)N
    C1 S3:0 O2:1 N4:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    Molecule: C[C@@](S)(O)N
    C1 S2:0 O3:1 N4:2 C0:3 (CHI_TETRAHEDRAL_CCW)

    Assign neighbor ranks to molecules with a stereogenic double bond:

    >>> for smiles in [r"CC(/Cl)=C(N)/C", r"CC(/Cl)=C(\\N)C"]:
    ...     print(f"Molecule: {{smiles}}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol, force=True)
    ...     print(describe_neighbor_ranking(mol))
    Molecule: CC(/Cl)=C(N)/C
    C1 C3:0 Cl2:1 C0:2 (STEREOTRANS)
    C3 C1:0 N4:1 C5:2 (STEREOTRANS)
    Molecule: CC(/Cl)=C(\\N)C
    C1 C3:0 Cl2:1 C0:2 (STEREOTRANS)
    C3 C1:0 N4:1 C5:2 (STEREOTRANS)

    """
    if not force and mol.HasProp(HAS_NEIGHBOR_RANKS) and mol.GetBoolProp(HAS_NEIGHBOR_RANKS):
        return

    atom_ranks = CanonicalRankAtoms(
        mol, breakTies=False, includeChirality=False, includeAtomMaps=False
    )
    atom_priorities = -np.fromiter(atom_ranks, dtype=int)

    chiral_centers = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() in {ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW}
    }

    cis_trans_bonds = [
        (bond, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.GetStereo()
        in {BondStereo.STEREOCIS, BondStereo.STEREOTRANS, BondStereo.STEREOZ, BondStereo.STEREOE}
    ]

    exceptions = {
        index: None for index in (range(mol.GetNumAtoms()) if all_atoms else chiral_centers)
    }
    for _, begin, end in cis_trans_bonds:
        exceptions[begin] = end
        exceptions[end] = begin

    second_neighbor = {}
    for index, exception in exceptions.items():
        atom = mol.GetAtomWithIdx(index)
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != exception]
        neighbor_priorities = atom_priorities[np.array(neighbors)]
        ranks = np.searchsorted(np.sort(neighbor_priorities), neighbor_priorities)
        if exception is not None:
            neighbors.append(exception)
            ranks = np.append(ranks + 1, 0)
        for nbr, rank in zip(neighbors, ranks):
            bond = mol.GetBondBetweenAtoms(index, nbr)
            bond.SetIntProp(END_RANK if bond.GetBeginAtomIdx() == index else BEGIN_RANK, int(rank))
            if exception is not None and rank == 1:
                second_neighbor[index] = nbr

        # Handle tetrahedral chiral centers
        if index in chiral_centers:
            atom.SetBoolProp(FLIP_CHIRAL_TAG, is_odd_permutation(*ranks))

    # Handle cis-trans double bonds
    for bond, begin, end in cis_trans_bonds:
        left, right = bond.GetStereoAtoms()
        bond.SetBoolProp(
            FLIP_STEREO, bool((second_neighbor[begin] == left) != (second_neighbor[end] == right))
        )

    mol.SetBoolProp(HAS_NEIGHBOR_RANKS, True)


def describe_neighbor_ranking(mol: Mol, include_leaves: bool = False) -> str:
    """Return a string representation of the neighbor ranking for all atoms in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule for which the neighbor ranking strings are generated.

    Returns
    -------
    str
        A string that lists the neighbor ranking strings for all atoms in the molecule.

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C[C@](O)(/C(N)=C/O)N")
    >>> print(describe_neighbor_ranking(mol))
    Molecule does not have neighbor ranks.
    >>> assign_neighbor_ranking(mol)
    >>> print(describe_neighbor_ranking(mol))
    C1 C3:0 O2:1 N7:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    C3 C5:0 C1:1 N4:2 (STEREOTRANS)
    C5 C3:0 O6:1 (STEREOTRANS)
    >>> mol = Chem.AddHs(mol)
    >>> assign_neighbor_ranking(mol, all_atoms=True, force=True)
    >>> print(describe_neighbor_ranking(mol))
    C1 C0:0 N7:1 C3:2 O2:3 (CHI_TETRAHEDRAL_CW)
    C0 C1:0 H8:1 H9:1 H10:1
    O2 C1:0 H11:1
    C3 C5:0 C1:1 N4:2 (STEREOTRANS)
    N4 C3:0 H12:1 H13:1
    C5 C3:0 O6:1 H14:2 (STEREOTRANS)
    O6 C5:0 H15:1
    N7 C1:0 H16:1 H17:1

    """
    if not (mol.HasProp(HAS_NEIGHBOR_RANKS) and mol.GetBoolProp(HAS_NEIGHBOR_RANKS)):
        return "Molecule does not have neighbor ranks."

    neighbors = {}
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bond.HasProp(BEGIN_RANK):
            rank = bond.GetIntProp(BEGIN_RANK)
            neighbors.setdefault(end, []).append((begin, rank))
        if bond.HasProp(END_RANK):
            rank = bond.GetIntProp(END_RANK)
            neighbors.setdefault(begin, []).append((end, rank))

    def atom_str(atom_idx: Atom) -> str:
        atom = mol.GetAtomWithIdx(atom_idx)
        return f"{atom.GetSymbol()}{atom.GetIdx()}"

    lines = {}
    for atom_idx, neighbors in neighbors.items():
        atom = mol.GetAtomWithIdx(atom_idx)
        neighbors.sort(key=lambda x: x[1])
        output = [atom_str(atom_idx)]
        for neighbor, rank in neighbors:
            output.append(f"{atom_str(neighbor)}:{rank}")
        tag = get_canonical_chiral_tag(atom)
        if tag != ChiralType.CHI_UNSPECIFIED:
            output.append(f"({tag.name})")
        lines[atom_idx] = output
    for bond in mol.GetBonds():
        tag = get_canonical_stereo(bond)
        if tag != BondStereo.STEREONONE:
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            lines[begin].append(f"({tag.name})")
            lines[end].append(f"({tag.name})")
    too_few = 2 - int(include_leaves)
    return "\n".join([" ".join(line) for line in lines.values() if len(line) > too_few])


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
    flip = atom.HasProp(FLIP_CHIRAL_TAG) and atom.GetBoolProp(FLIP_CHIRAL_TAG)
    if (chiral_tag == ChiralType.CHI_TETRAHEDRAL_CW) == flip:
        return ChiralType.CHI_TETRAHEDRAL_CCW
    return ChiralType.CHI_TETRAHEDRAL_CW


def get_canonical_stereo(bond: Bond) -> BondStereo:
    r"""Return the canonical stereochemistry flag of a bond.

    .. note::
        This function never returns ``STEREOZ`` or ``STEREOE`` flags. Instead, it returns
        ``STEREOCIS`` or ``STEREOTRANS``, respectively.

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
    C1 C3:0 Cl2:1 C0:2 (STEREOTRANS)
    C3 C1:0 N4:1 C5:2 (STEREOTRANS)
    Molecule: CC(/Cl)=C(\N)F
    C1 C3:0 Cl2:1 C0:2 (STEREOCIS)
    C3 C1:0 F5:1 N4:2 (STEREOCIS)

    """
    stereo = bond.GetStereo()
    if stereo not in {
        BondStereo.STEREOCIS,
        BondStereo.STEREOTRANS,
        BondStereo.STEREOZ,
        BondStereo.STEREOE,
    }:
        return stereo
    flip = bond.HasProp(FLIP_STEREO) and bond.GetBoolProp(FLIP_STEREO)
    if (stereo in {BondStereo.STEREOCIS, BondStereo.STEREOZ}) == flip:
        return BondStereo.STEREOTRANS
    return BondStereo.STEREOCIS


def get_begin_rank(bond: Bond) -> int:
    f"""Return the rank of the begin atom among the end atom's neighbors.

    If the bond does not have a ``{BEGIN_RANK}`` property, return -1.

    Parameters
    ----------
    bond : Chem.Bond
        The bond whose begin atom's rank is to be returned.

    Returns
    -------
    int
        The rank of the begin atom among the end atom's neighbors.
    """
    if bond.HasProp(BEGIN_RANK):
        return bond.GetIntProp(BEGIN_RANK)
    return -1


def get_end_rank(bond: Bond) -> int:
    f"""Return the rank of the begin atom among the end atom's neighbors.

    If the bond does not have a ``{END_RANK}`` property, return -1.

    Parameters
    ----------
    bond : Chem.Bond
        The bond whose end atom's rank is to be returned.

    Returns
    -------
    int
        The rank of the end atom among the begin atom's neighbors.
    """
    if bond.HasProp(END_RANK):
        return bond.GetIntProp(END_RANK)
    return -1
