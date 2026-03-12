import numpy as np
from rdkit.Chem.rdchem import Atom, BondStereo, ChiralType, Mol
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms


def is_odd_permutation(i: int, j: int, k: int, m: int | None = None) -> bool:
    r"""Return whether the permutation parity is odd.

    Parameters
    ----------
    i, j, k : int
        Priority tags of three neighbors.
    m : int | None, default=None
        Optional priority tag of a fourth neighbor.

    Returns
    -------
    bool
        ``True`` when the number of pairwise inversions is odd; otherwise ``False``.

    Notes
    -----
    This helper function should be used for tetrahedral chiral centers only. It requires
    all neighbor priority tags to be unique. If only three neighbors are specified, the fourth
    neighbor is assumed to be an implicit hydrogen, whose priority would be the lowest
    if it were explicitly present.
    """
    swaps = int(i > j) + int(i > k) + int(j > k)
    if m is not None:
        swaps += int(i > m) + int(j > m) + int(k > m)
    return bool(swaps % 2)


def has_minimum_value(key: int, among: dict[int, int], excluding: int) -> bool:
    """Return whether a given key has the minimum value among those in a dictionary, excluding
    another key from the dictionary.

    Assumes the given key is present in the dictionary.

    Parameters
    ----------
    key : int
        The key whose value is to be checked.
    among : dict[int, int]
        The dictionary of values to check.
    excluding : int
        The key to exclude from consideration.

    Returns
    -------
    bool
        ``True`` if the key has the minimum value, ``False`` otherwise.
    """
    value = among[key]
    return all(v >= value for k, v in among.items() if k != excluding)


def normalize_stereo(stereo: BondStereo) -> BondStereo:
    """Return the normalized stereochemistry flag of a bond."""
    if stereo == BondStereo.STEREOZ:
        return BondStereo.STEREOCIS
    if stereo == BondStereo.STEREOE:
        return BondStereo.STEREOTRANS
    return stereo


def flip_stereo(stereo: BondStereo) -> BondStereo:
    """Return the flipped stereochemistry flag of a bond."""
    if stereo in {BondStereo.STEREOCIS, BondStereo.STEREOZ}:
        return BondStereo.STEREOTRANS
    if stereo in {BondStereo.STEREOTRANS, BondStereo.STEREOE}:
        return BondStereo.STEREOCIS
    return stereo


def flip_chiral_tag(chiral_tag: ChiralType) -> ChiralType:
    """Return the flipped chiral tag of an atom."""
    if chiral_tag == ChiralType.CHI_TETRAHEDRAL_CW:
        return ChiralType.CHI_TETRAHEDRAL_CCW
    if chiral_tag == ChiralType.CHI_TETRAHEDRAL_CCW:
        return ChiralType.CHI_TETRAHEDRAL_CW
    return chiral_tag


def mol_with_neighbor_priority_tags(mol: Mol) -> Mol:
    r"""Return a copy of a molecule with neighbor priority information.

    This function derives local neighbor priorities from RDKit canonical atom
    ranks (:func:`~rdkit.Chem.CanonicalRankAtoms`, with ``breakTies`` and
    ``includeAtomMaps`` set to ``False``).

    Given all neighbors of an atom, the priority tag of a neighbor equals the
    number of other neighbors with strictly higher canonical atom ranks.

    For each bond in the molecule, the function stores the neighbor priority
    tags seen from each endpoint:

    - ``endAtomPriorityTag``: priority of the end atom among neighbors of the
      begin atom.
    - ``beginAtomPriorityTag``: priority of the begin atom among neighbors of
      the end atom.

    If necessary:

    - Atom chiral tags are flipped (CW ↔ CCW) to match the local neighbor
      priority order.
    - Bond stereo flags are normalized (Z → cis, E → trans) and flipped
      (cis ↔ trans) so that the stereo annotation is consistent with the
      local neighbor priority order.

    Finally, the function adds a boolean property
    ``hasNeighborPriorityTags=True`` to the molecule.

    Ties in canonical atom ranks are allowed and reflect molecular symmetry.
    In well-formed molecules, all neighbors of a tetrahedral stereocenter
    have unique canonical ranks. Similarly, substituents on each side of a
    stereo-tagged double bond have unique canonical ranks.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule whose bonds and atoms are annotated.

    Returns
    -------
    rdkit.Chem.Mol
        A copy of the molecule with annotated neighbor priority tags and
        stereo annotations adjusted to the local neighbor priority order.
    """
    mol = Mol(mol)
    if mol.HasProp("hasNeighborPriorityTags") and mol.GetBoolProp("hasNeighborPriorityTags"):
        return mol

    atom_ranks = CanonicalRankAtoms(mol, breakTies=False, includeAtomMaps=False)
    atom_priorities = -np.fromiter(atom_ranks, dtype=int)

    # For each atom index, map neighbor index -> priority tag.
    neighbors: list[dict[int, int]] = []

    for atom in mol.GetAtoms():
        neighbor_indices = np.fromiter((atom.GetIdx() for atom in atom.GetNeighbors()), dtype=int)
        neighbor_priorities = atom_priorities[neighbor_indices]

        # Tag = number of neighbors with strictly higher canonical atom rank.
        tags = np.searchsorted(np.sort(neighbor_priorities), neighbor_priorities)
        neighbors.append(dict(zip(neighbor_indices, tags)))

        chiral_tag = atom.GetChiralTag()
        if chiral_tag in {ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW}:
            if is_odd_permutation(*tags):
                atom.SetChiralTag(flip_chiral_tag(chiral_tag))

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endAtomPriorityTag", int(neighbors[begin][end]))
        bond.SetIntProp("beginAtomPriorityTag", int(neighbors[end][begin]))

        stereo = bond.GetStereo()
        if stereo in {
            BondStereo.STEREOCIS,
            BondStereo.STEREOTRANS,
            BondStereo.STEREOZ,
            BondStereo.STEREOE,
        }:
            left, right = bond.GetStereoAtoms()
            left_has_highest_priority = has_minimum_value(
                left, among=neighbors[begin], excluding=end
            )
            right_has_highest_priority = has_minimum_value(
                right, among=neighbors[end], excluding=begin
            )
            if left_has_highest_priority != right_has_highest_priority:
                bond.SetStereo(flip_stereo(stereo))
            elif stereo in {BondStereo.STEREOZ, BondStereo.STEREOE}:
                bond.SetStereo(normalize_stereo(stereo))

    mol.SetBoolProp("hasNeighborPriorityTags", True)
    return mol


def describe_neighbor_tagging(mol: Mol, include_leaves: bool = False) -> str:
    """Return a textual description of neighbor priority tags for all atoms.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule for which the neighbor priority-tag description is generated.
    include_leaves : bool, default=False
        Whether to include leaves (i.e., atoms with only one neighbor) in the
        priority-tag description.

    Returns
    -------
    str
        A string listing neighbor priority tags for all atoms in the molecule.

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")
    >>> print(describe_neighbor_tagging(mol))
    Molecule does not have neighbor priority tags.
    >>> mol = mol_with_neighbor_priority_tags(mol)
    >>> print(describe_neighbor_tagging(mol))
    C1 C3:0 O2:1 N6:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    C3 C1:0 C4:1
    C4 C3:0 O5:1
    C3-C4 STEREOTRANS
    >>> print(describe_neighbor_tagging(mol, include_leaves=True))
    C0 C1:0
    C1 C3:0 O2:1 N6:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    O2 C1:0
    C3 C1:0 C4:1
    C4 C3:0 O5:1
    O5 C4:0
    N6 C1:0
    C3-C4 STEREOTRANS

    """
    if not (mol.HasProp("hasNeighborPriorityTags") and mol.GetBoolProp("hasNeighborPriorityTags")):
        return "Molecule does not have neighbor priority tags."

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
                    rank = bond.GetIntProp("endAtomPriorityTag")
                else:
                    neighbor = begin
                    rank = bond.GetIntProp("beginAtomPriorityTag")
                neighbors.append((neighbor, rank))
            neighbors.sort(key=lambda x: x[1])
            output = [f"{atom.GetSymbol()}{atom_idx}"]
            for neighbor, rank in neighbors:
                output.append(f"{atom_str(neighbor)}:{rank}")
            tag = atom.GetChiralTag()
            if tag != ChiralType.CHI_UNSPECIFIED:
                output.append(f"({tag.name})")
            lines.append(" ".join(output))
    for bond in mol.GetBonds():
        tag = bond.GetStereo()
        if tag != BondStereo.STEREONONE:
            begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
            lines.append(f"{atom_str(begin)}-{atom_str(end)} {tag.name}")
    return "\n".join(lines)
