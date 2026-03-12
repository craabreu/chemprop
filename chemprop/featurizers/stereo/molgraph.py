import numpy as np
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.stereo.neighbor_tagging import mol_with_neighbor_priority_tags


NUM_NEIGHBOR_TAG_BITS = 4


class StereoMoleculeMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    """Featurizes molecules with asymmetric stereochemical information.

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    def __post_init__(self):
        super().__post_init__()
        self.bond_fdim += NUM_NEIGHBOR_TAG_BITS

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        mol = mol_with_neighbor_priority_tags(mol)
        num_extra_bond_feats = (
            len(bond_features_extra)
            if bond_features_extra is not None
            else mol.GetNumBonds()
        )
        placeholder = np.zeros(
            (num_extra_bond_feats, NUM_NEIGHBOR_TAG_BITS), dtype=np.single
        )
        if bond_features_extra is None:
            bond_features_extra = placeholder
        else:
            bond_features_extra = np.concatenate(
                (placeholder, bond_features_extra), axis=1
            )
        mol_graph = super().__call__(mol, atom_features_extra, bond_features_extra)
        start = len(self.bond_featurizer)
        for bond_idx, bond in enumerate(mol.GetBonds()):
            begin_tag = int(bond.GetIntProp("beginAtomPriorityTag"))
            end_tag = int(bond.GetIntProp("endAtomPriorityTag"))
            forward_row = 2 * bond_idx
            reverse_row = forward_row + 1
            mol_graph.E[forward_row, start + min(end_tag, 3)] = 1.0
            mol_graph.E[reverse_row, start + min(begin_tag, 3)] = 1.0
        return mol_graph
