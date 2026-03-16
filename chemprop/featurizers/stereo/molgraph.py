from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.stereo.neighbor_tagging import (
    CHIRAL_CENTER_TAGS,
    mol_with_neighbor_priority_tags,
    normalize_chiral_tags_to_ccw,
)

DEFAULT_NUM_NEIGHBOR_BITS: int = 4
HOT_ONE: np.single = np.single(1.0)


@dataclass
class StereoMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    """Featurize molecules with asymmetric stereochemical edge information.

    This featurizer extends :class:`SimpleMoleculeMolGraphFeaturizer` by adding a number ``n``
    of neighbor-priority bits to each directed bond feature vector. The inserted bits encode a
    neighbor-priority ordering derived from RDKit's :func:`~rdkit.Chem.CanonicalRankAtoms`
    function with arguments ``breakTies`` and ``includeAtomMaps`` set to ``False`` (see [1]_).

    Neighbor priorities are encoded as 0-based tags, where tag 0 denotes the highest-priority
    neighbor, tag 1 the second-highest, and so on. Tags greater than or equal to ``n`` are
    clipped to the last bit.

    | Tag value | Meaning                   | One-hot encoding  |
    |-----------|---------------------------|-------------------|
    | 0         | highest priority neighbor | [1, 0, 0, ..., 0] |
    | 1         | second-highest            | [0, 1, 0, ..., 0] |
    | ...       | ...                       | ...               |
    | >= n - 1  | clamped to last bit       | [0, 0, 0, ..., 1] |

    The resulting bond feature layout is:

    ``[base bond features | n neighbor-priority bits | user-provided extra bond features]``.

    The priorities of the neighbors of an atom are encoded either in the edges that converge
    to that atom (convergent mode) or in the edges that originate from it (divergent mode).
    In standard Chemprop featurization, the two directed edges associated with a bond initially
    carry identical features. Here, the inserted neighbor-priority bits can differ between the
    two directions already at featurization time.

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        Featurizer used to compute atom features.
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        Featurizer used to compute base bond features.
    extra_atom_fdim : int, default=0
        Dimension of user-provided extra atom features concatenated to each atom row.
    extra_bond_fdim : int, default=0
        Dimension of user-provided extra bond features concatenated after the inserted
        neighbor-priority bits.
    num_neighbor_bits : int, default=4
        Number of neighbor-priority bits to encode in each directed edge.
    stereo_atoms_only : bool, default=True
        If ``True``, encode neighbor-priority bits only for bonds adjacent to stereochemically
        relevant atoms (tetrahedral chiral centers).
        If ``False``, all atoms are eligible.
    normalize_chiral_tags : bool, default=True
        If ``True``, transforms all ``CHI_TETRAHEDRAL_CW`` chiral tags to
        ``CHI_TETRAHEDRAL_CCW`` by exchanging local priority tags 0 and 1.
        This transfers the distinction between enantiomers from graph nodes to directed edges.
    convergent_mode : bool, default=True
        Controls which directed edge receives a target atom's neighbor-priority tag.
        If ``True``, tags are written on directed edges that converge to the target atom.
        If ``False``, the same tags are written on the opposite directed edges (divergent mode).

    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.CanonicalRankAtoms
    """

    stereo_atoms_only: bool = True
    normalize_chiral_tags: bool = True
    convergent_mode: bool = True
    num_neighbor_bits: int = DEFAULT_NUM_NEIGHBOR_BITS

    def __post_init__(self):
        super().__post_init__()
        if self.num_neighbor_bits < 1:
            raise ValueError("num_neighbor_bits must be >= 1")
        self.bond_fdim += self.num_neighbor_bits

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        num_extra_bond_feats = (
            len(bond_features_extra) if bond_features_extra is not None else mol.GetNumBonds()
        )
        placeholder = np.zeros((num_extra_bond_feats, self.num_neighbor_bits), dtype=np.single)
        if bond_features_extra is None:
            bond_features_extra = placeholder
        else:
            bond_features_extra = np.concatenate((placeholder, bond_features_extra), axis=1)

        target_nodes = self._get_atoms_to_encode(mol)  # encoded edges converge to these nodes
        if not target_nodes:
            return super().__call__(mol, atom_features_extra, bond_features_extra)

        mol_with_tags = mol_with_neighbor_priority_tags(mol)
        if self.normalize_chiral_tags:
            normalize_chiral_tags_to_ccw(mol_with_tags)
        mol_graph = super().__call__(mol_with_tags, atom_features_extra, bond_features_extra)

        start = len(self.bond_featurizer)
        max_tag = self.num_neighbor_bits - 1
        for bond_idx, bond in enumerate(mol_with_tags.GetBonds()):
            source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            forward_row = 2 * bond_idx
            reverse_row = forward_row + 1
            if target in target_nodes:
                source_tag = int(bond.GetIntProp("beginAtomPriorityTag"))
                row = forward_row if self.convergent_mode else reverse_row
                mol_graph.E[row, start + min(source_tag, max_tag)] = HOT_ONE
            if source in target_nodes:
                target_tag = int(bond.GetIntProp("endAtomPriorityTag"))
                row = reverse_row if self.convergent_mode else forward_row
                mol_graph.E[row, start + min(target_tag, max_tag)] = HOT_ONE
        return mol_graph

    def _get_atoms_to_encode(self, mol: Chem.Mol) -> set[int]:
        if self.stereo_atoms_only:
            return {
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetChiralTag() in CHIRAL_CENTER_TAGS
            }
        return {atom.GetIdx() for atom in mol.GetAtoms()}
