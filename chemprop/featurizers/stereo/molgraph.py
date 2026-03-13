from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.stereo.neighbor_tagging import (
    CHIRAL_CENTER_TAGS,
    STEREOGENIC_BOND_TAGS,
    mol_with_neighbor_priority_tags,
    normalize_chiral_tags_to_ccw,
)

NUM_NEIGHBOR_TAG_BITS: int = 4
HOT_ONE: np.single = np.single(1.0)


@dataclass
class StereoMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
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
    stereo_atoms_only : bool, default=True
        whether to encode neighbor-tag one-hots only on directed edges that converge to
        stereochemistry-relevant atoms (chiral centers and atoms on stereo-tagged bonds)
    normalize_chiral_tags : bool, default=True
        whether to normalize tetrahedral chiral tags to CCW (swapping the two top-priority
        neighbors, if necessary) before featurization
    """

    stereo_atoms_only: bool = True
    normalize_chiral_tags: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.bond_fdim += NUM_NEIGHBOR_TAG_BITS

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        num_extra_bond_feats = (
            len(bond_features_extra) if bond_features_extra is not None else mol.GetNumBonds()
        )
        placeholder = np.zeros((num_extra_bond_feats, NUM_NEIGHBOR_TAG_BITS), dtype=np.single)
        if bond_features_extra is None:
            bond_features_extra = placeholder
        else:
            bond_features_extra = np.concatenate((placeholder, bond_features_extra), axis=1)

        atoms_to_encode = self._get_atoms_to_encode(mol)
        if not atoms_to_encode:
            return super().__call__(mol, atom_features_extra, bond_features_extra)

        mol_with_tags = mol_with_neighbor_priority_tags(mol)
        if self.normalize_chiral_tags:
            normalize_chiral_tags_to_ccw(mol_with_tags)
        mol_graph = super().__call__(mol_with_tags, atom_features_extra, bond_features_extra)

        start = len(self.bond_featurizer)
        max_tag = NUM_NEIGHBOR_TAG_BITS - 1
        for bond in mol_with_tags.GetBonds():
            bond_idx = bond.GetIdx()
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if end_idx in atoms_to_encode:
                end_tag = int(bond.GetIntProp("endAtomPriorityTag"))
                forward_row = 2 * bond_idx
                mol_graph.E[forward_row, start + min(end_tag, max_tag)] = HOT_ONE
            if begin_idx in atoms_to_encode:
                begin_tag = int(bond.GetIntProp("beginAtomPriorityTag"))
                reverse_row = 2 * bond_idx + 1
                mol_graph.E[reverse_row, start + min(begin_tag, max_tag)] = HOT_ONE
        return mol_graph

    def _get_atoms_to_encode(self, mol: Chem.Mol) -> set[int]:
        if self.stereo_atoms_only:
            atoms_to_encode = {
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetChiralTag() in CHIRAL_CENTER_TAGS
            }
            for bond in mol.GetBonds():
                if bond.GetStereo() in STEREOGENIC_BOND_TAGS:
                    atoms_to_encode.add(bond.GetBeginAtomIdx())
                    atoms_to_encode.add(bond.GetEndAtomIdx())
            return atoms_to_encode
        return {atom.GetIdx() for atom in mol.GetAtoms()}
