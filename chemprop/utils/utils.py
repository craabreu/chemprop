from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Iterator

import numpy as np
from rdkit import Chem


class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name: str | EnumMapping) -> EnumMapping:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {cls.keys()}"
            )

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())


def make_mol(smi: str, keep_h: bool, add_h: bool, ignore_chirality: bool = False) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        If True, adds hydrogens to the molecule.
    ignore_chirality : bool, optional
        If True, ignores chirality information when constructing the molecule. Default is False.

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        )
    else:
        mol = Chem.MolFromSmiles(smi)

    if mol is None:
        raise RuntimeError(f"SMILES {smi} is invalid! (RDKit returned None)")

    if add_h:
        mol = Chem.AddHs(mol)

    if ignore_chirality:
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    return mol


def is_odd_permutation(i: int, j: int, k: int, m: int | None = None) -> int:
    r"""Check if a permutation is odd."""
    swaps = int(i > j) + int(i > k) + int(j > k)
    if m is not None:
        swaps += int(i > m) + int(j > m) + int(k > m)
    return bool(swaps % 2)


CHIRAL_CENTER_TAGS = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}

STEREOGENIC_BOND_TAGS = {
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
}


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
    >>> from rdkit import Chem
    >>> import numpy as np

    Define a function for printing bond rankings

    >>> def describe_atom(atom):
    ...     return f"{atom.GetSymbol()}{atom.GetIdx()}"
    >>> def print_rankings(mol):
    ...     ranks = {describe_atom(atom): [] for atom in mol.GetAtoms()}
    ...     for bond in mol.GetBonds():
    ...         begin = describe_atom(bond.GetBeginAtom())
    ...         end = describe_atom(bond.GetEndAtom())
    ...         ranks[begin].append((end, bond.GetIntProp("endRankFromBegin")))
    ...         ranks[end].append((begin, bond.GetIntProp("beginRankFromEnd")))
    ...     for atom in ranks:
    ...         if len(ranks[atom]) > 1:
    ...             print(atom, *sorted(ranks[atom], key=lambda x: x[1]))

    Assign neighbor ranks to a molecule

    >>> mol = Chem.MolFromSmiles("NC(O)=C(OC)OC")
    >>> assign_neighbor_ranking(mol)
    >>> print_rankings(mol)
    C1 ('C3', 0) ('O2', 1) ('N0', 2)
    C3 ('C1', 0) ('O4', 1) ('O6', 1)
    O4 ('C3', 0) ('C5', 1)
    O6 ('C3', 0) ('C7', 1)

    Assign neighbor ranks to molecules with a tetrahedral chiral center

    >>> for smiles in ["C[C@](O)(S)N", "C[C@@](S)(O)N"]:
    ...     print(f"\nMolecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print_rankings(mol)
    ...     print("Flip tag:", mol.GetAtomWithIdx(1).GetBoolProp("flipChiralTag"))
    <BLANKLINE>
    Molecule: C[C@](O)(S)N
    C1 ('S3', 0) ('O2', 1) ('N4', 2) ('C0', 3)
    Flip tag: False
    <BLANKLINE>
    Molecule: C[C@@](S)(O)N
    C1 ('S2', 0) ('O3', 1) ('N4', 2) ('C0', 3)
    Flip tag: True

    Assign neighbor ranks to molecules with a stereogenic double bond:

    >>> for smiles in [r"CC(/Cl)=C(N)/C", r"CC(/Cl)=C(\N)C"]:
    ...     print(f"\nMolecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print_rankings(mol)
    ...     for set_from_directions in [False, True]:
    ...         if set_from_directions:
    ...             Chem.SetBondStereoFromDirections(mol)
    ...             assign_neighbor_ranking(mol, force=True)
    ...         bond = mol.GetBondWithIdx(2)
    ...         print("Originally:", bond.GetStereo().name, *bond.GetStereoAtoms())
    ...         print("Flip stereo:", bond.GetBoolProp("flipStereo"))
    <BLANKLINE>
    Molecule: CC(/Cl)=C(N)/C
    C1 ('C3', 0) ('Cl2', 1) ('C0', 2)
    C3 ('C1', 0) ('N4', 1) ('C5', 2)
    Originally: STEREOE 2 4
    Flip stereo: False
    Originally: STEREOCIS 2 5
    Flip stereo: True
    <BLANKLINE>
    Molecule: CC(/Cl)=C(\N)C
    C1 ('C3', 0) ('Cl2', 1) ('C0', 2)
    C3 ('C1', 0) ('N4', 1) ('C5', 2)
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

        if atom.GetChiralTag() in CHIRAL_CENTER_TAGS:
            atom.SetBoolProp("flipChiralTag", is_odd_permutation(*ranks))

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endRankFromBegin", int(sorted_neighbors[begin][end]))
        bond.SetIntProp("beginRankFromEnd", int(sorted_neighbors[end][begin]))

        if bond.GetStereo() in STEREOGENIC_BOND_TAGS:
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


def pretty_shape(shape: Iterable[int]) -> str:
    """Make a pretty string from an input shape

    Example
    --------
    >>> X = np.random.rand(10, 4)
    >>> X.shape
    (10, 4)
    >>> pretty_shape(X.shape)
    '10 x 4'
    """
    return " x ".join(map(str, shape))
