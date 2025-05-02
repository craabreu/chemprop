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


def set_relative_neighbor_ranking(mol: Chem.Mol, force: bool = False) -> None:
    r"""Add neighbor ranking information to the bonds of a molecule.

    Neighbors of each atom are sorted in descending order based on their canonical
    ranks (see :rdmolfiles:`CanonicalRankAtoms`). The descending order keeps hydrogens
    at the end of the list.

    Two integer properties are added to each bond in the molecule:

    - `endRankFromBegin`: The relative rank of the bond's end atom with respect to all
        neighbors of the bond's begin atom.
    - `beginRankFromEnd`: The relative rank of the bond's begin atom with respect to all
        neighbors of the bond's end atom.

    A relative rank is the position in a sorted list of canonical ranks in descending
    order. For example, if an atom has three neighbors with canonical ranks `(3, 1, 7)`,
    their relative ranks are `(1, 2, 0)`. Tied canonical ranks result in the same
    relative rank, corresponding to the least position they occupy in the descending
    sorted list.

    The molecule is tagged with a boolean property `hasNeighborRanks` set to True.

    Parameters
    ----------
    mol
        The molecule to add neighbor ranking information to.
    force
        Whether to add neighbor ranking information even if it has already been added
        (default is False).

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
    >>> set_relative_neighbor_ranking(mol)
    >>> print_rankings(mol)
    C1 ('C3', 0) ('O2', 1) ('N0', 2)
    C3 ('C1', 0) ('O4', 1) ('O6', 1)
    O4 ('C3', 0) ('C5', 1)
    O6 ('C3', 0) ('C7', 1)

    Assign neighbor ranks to molecules with a tetrahedral chiral center

    >>> for smiles in ["C[C@](O)(S)N", "C[C@@](S)(O)N"]:
    ...     print(f"\nMolecule: {smiles}")
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     set_relative_neighbor_ranking(mol)
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
    ...     set_relative_neighbor_ranking(mol)
    ...     print_rankings(mol)
    ...     bond = mol.GetBondWithIdx(2)
    ...     print(
    ...         "Originally:",
    ...         ["Trans", "Cis"][bond.GetStereo() == Chem.BondStereo.STEREOCIS],
    ...         *bond.GetStereoAtoms()
    ...     )
    ...     print("Flip stereo:", bond.GetBoolProp("flipStereo"))
    <BLANKLINE>
    Molecule: CC(/Cl)=C(N)/C
    C1 ('C3', 0) ('Cl2', 1) ('C0', 2)
    C3 ('C1', 0) ('N4', 1) ('C5', 2)
    Originally: Cis 2 5
    Flip stereo: True
    <BLANKLINE>
    Molecule: CC(/Cl)=C(\N)C
    C1 ('C3', 0) ('Cl2', 1) ('C0', 2)
    C3 ('C1', 0) ('N4', 1) ('C5', 2)
    Originally: Trans 2 4
    Flip stereo: False

    """
    if not force and mol.HasProp("hasNeighborRanks"):
        return
    all_priorities = -np.fromiter(
        Chem.CanonicalRankAtoms(
            mol, breakTies=False, includeChirality=False, includeAtomMaps=False
        ),
        dtype=int,
    )
    Chem.SetBondStereoFromDirections(mol)
    sorted_neighbors = []
    for atom in mol.GetAtoms():
        neighbors = np.fromiter((atom.GetIdx() for atom in atom.GetNeighbors()), dtype=int)
        neighbor_priorities = all_priorities[neighbors]
        ranks = np.searchsorted(np.sort(neighbor_priorities), neighbor_priorities)
        sorted_neighbors.append(dict(zip(neighbors, ranks)))

        # Handle tetrahedral stereocenters
        if atom.GetChiralTag() in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ):
            atom.SetBoolProp("flipChiralTag", is_odd_permutation(*ranks))

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endRankFromBegin", int(sorted_neighbors[begin][end]))
        bond.SetIntProp("beginRankFromEnd", int(sorted_neighbors[end][begin]))

        # Handle cis/trans stereobonds
        if bond.GetStereo() in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
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
