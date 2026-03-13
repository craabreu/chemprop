import numpy as np
from rdkit.Chem.rdchem import Atom, BondStereo, ChiralType, Mol
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms

CHIRAL_CENTER_TAGS = {ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW}
STEREOGENIC_BOND_TAGS = {
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
}


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


def top_priority_neighbor(neighbors: dict[int, int], excluding: int) -> int:
    """Return the top-priority neighbor, excluding a given neighbor from consideration.

    Returns
    -------
    int
        The top-priority neighbor, except for the excluded neighbor.
    """
    subset = {k: v for k, v in neighbors.items() if k != excluding}
    return min(subset.keys(), key=subset.get)


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

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C[C@H](O)N")
    >>> tagged = mol_with_neighbor_priority_tags(mol)
    >>> tagged.GetBoolProp("hasNeighborPriorityTags")
    True
    >>> center = next(
    ...     atom
    ...     for atom in tagged.GetAtoms()
    ...     if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    ... )
    >>> center.GetChiralTag().name
    'CHI_TETRAHEDRAL_CCW'
    >>> describe_neighbor_tagging(tagged)
    'C1 O2:0 N3:1 C0:2 (CHI_TETRAHEDRAL_CCW)'
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
        if chiral_tag in CHIRAL_CENTER_TAGS:
            if is_odd_permutation(*tags):
                atom.SetChiralTag(flip_chiral_tag(chiral_tag))

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endAtomPriorityTag", int(neighbors[begin][end]))
        bond.SetIntProp("beginAtomPriorityTag", int(neighbors[end][begin]))

        stereo = bond.GetStereo()
        if stereo in STEREOGENIC_BOND_TAGS:
            stereo_left, stereo_right = bond.GetStereoAtoms()
            top_left = top_priority_neighbor(neighbors[begin], end)
            top_right = top_priority_neighbor(neighbors[end], begin)
            if (stereo_left == top_left) != (stereo_right == top_right):
                bond.SetStereo(flip_stereo(stereo))
            elif stereo in {BondStereo.STEREOZ, BondStereo.STEREOE}:
                bond.SetStereo(normalize_stereo(stereo))
            bond.SetStereoAtoms(int(top_left), int(top_right))

    mol.SetBoolProp("hasNeighborPriorityTags", True)
    return mol


def normalize_chiral_tags_to_ccw(mol: Mol) -> None:
    """Modify a molecule in place by normalizing tetrahedral chiral tags to CCW.

    This function converts all tetrahedral stereocenters tagged as
    ``CHI_TETRAHEDRAL_CW`` to ``CHI_TETRAHEDRAL_CCW`` while preserving the
    stereochemical meaning encoded by the local neighbor priority tags.

    The function assumes that the molecule has already been processed by
    :func:`mol_with_neighbor_priority_tags`.

    When a stereocenter is tagged ``CHI_TETRAHEDRAL_CW``, its orientation is
    reversed by swapping the priority tags of the two top-priority neighbors.
    This corresponds to reversing the permutation parity of the neighbor ordering.
    After this adjustment, the atom's chiral tag is set to ``CHI_TETRAHEDRAL_CCW``.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule whose tetrahedral chiral tags are normalized.

    Raises
    ------
    ValueError
        If the molecule does not have neighbor priority tags assigned.

    Example
    -------
    >>> from rdkit import Chem
    >>> raw = Chem.MolFromSmiles("C[C@H](O)N")
    >>> normalize_chiral_tags_to_ccw(raw)
    Traceback (most recent call last):
    ...
    ValueError: Molecule does not have neighbor priority tags.
    >>> mol = mol_with_neighbor_priority_tags(raw)
    >>> center = next(
    ...     atom
    ...     for atom in mol.GetAtoms()
    ...     if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    ... )
    >>> center.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)
    >>> normalize_chiral_tags_to_ccw(mol)
    >>> center.GetChiralTag().name
    'CHI_TETRAHEDRAL_CCW'
    """
    if not (mol.HasProp("hasNeighborPriorityTags") and mol.GetBoolProp("hasNeighborPriorityTags")):
        raise ValueError("Molecule does not have neighbor priority tags.")

    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW:
            idx = atom.GetIdx()
            for bond in atom.GetBonds():
                begin = bond.GetBeginAtomIdx()
                endpoint = "end" if begin == idx else "begin"
                label = f"{endpoint}AtomPriorityTag"
                tag = bond.GetIntProp(label)
                if tag < 2:
                    bond.SetIntProp(label, 1 - tag)

            atom.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CCW)


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
