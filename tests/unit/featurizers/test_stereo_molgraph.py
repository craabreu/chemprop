import numpy as np
import pytest
from rdkit import Chem

from chemprop.featurizers.stereo.molgraph import (
    NUM_NEIGHBOR_TAG_BITS,
    StereoMoleculeMolGraphFeaturizer,
)
from chemprop.featurizers.stereo.neighbor_tagging import mol_with_neighbor_priority_tags


def _encode_neighbor_tag(tag):
    one_hot = np.zeros(NUM_NEIGHBOR_TAG_BITS, dtype=np.single)
    one_hot[min(tag, 3)] = 1.0
    return one_hot


def test_stereo_bond_extra_features_are_appended_by_feature_dimension(mol):
    """Neighbor-tag one-hots and user bond extras are both appended to E."""
    n_bonds = mol.GetNumBonds()
    extra_bond_fdim = 3
    bond_features_extra = np.random.rand(n_bonds, extra_bond_fdim).astype(np.single)

    featurizer = StereoMoleculeMolGraphFeaturizer(extra_bond_fdim=extra_bond_fdim)
    mol_graph = featurizer(mol, bond_features_extra=bond_features_extra)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    for bond in tagged_mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin_tag = int(bond.GetIntProp("beginAtomPriorityTag"))
        end_tag = int(bond.GetIntProp("endAtomPriorityTag"))
        np.testing.assert_array_equal(mol_graph.E[2 * bond_idx, start:stop], _encode_neighbor_tag(end_tag))
        np.testing.assert_array_equal(
            mol_graph.E[2 * bond_idx + 1, start:stop], _encode_neighbor_tag(begin_tag)
        )

    np.testing.assert_allclose(mol_graph.E[::2, stop:], bond_features_extra)


def test_stereo_neighbor_tag_bits_can_be_asymmetric_between_edge_directions():
    """Directed edge features can differ when endpoint neighbor tags are different."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    featurizer = StereoMoleculeMolGraphFeaturizer()
    mol_graph = featurizer(mol)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    asymmetric_found = False
    for bond in tagged_mol.GetBonds():
        bond_idx = bond.GetIdx()
        if int(bond.GetIntProp("beginAtomPriorityTag")) != int(bond.GetIntProp("endAtomPriorityTag")):
            asymmetric_found = True
            forward = mol_graph.E[2 * bond_idx, start:stop]
            reverse = mol_graph.E[2 * bond_idx + 1, start:stop]
            assert not np.array_equal(forward, reverse)

    assert asymmetric_found


def test_stereo_neighbor_tag_bucket_for_tag_three_maps_to_last_bit():
    """Neighbor tag 3 is encoded in the last bucket bit (0001)."""
    mol = Chem.MolFromSmiles("C[C@](F)(Cl)Br")
    featurizer = StereoMoleculeMolGraphFeaturizer()
    mol_graph = featurizer(mol)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    found_tag_three = False
    for edge_idx, (src, dst) in enumerate(mol_graph.edge_index.T):
        src = int(src)
        dst = int(dst)
        bond = tagged_mol.GetBondBetweenAtoms(src, dst)
        assert bond is not None
        if src == bond.GetBeginAtomIdx():
            expected_tag = int(bond.GetIntProp("endAtomPriorityTag"))
        else:
            expected_tag = int(bond.GetIntProp("beginAtomPriorityTag"))

        if expected_tag == 3:
            found_tag_three = True
            np.testing.assert_array_equal(mol_graph.E[edge_idx, start:stop], np.array([0, 0, 0, 1]))

    assert found_tag_three


def test_stereo_neighbor_tag_bits_match_destination_tags_via_edge_index():
    """Tag one-hots on directed edges match destination neighbor tags from edge_index."""
    mol = Chem.MolFromSmiles("C[C@](F)(Cl)Br")
    featurizer = StereoMoleculeMolGraphFeaturizer()
    mol_graph = featurizer(mol)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    for edge_idx, (src, dst) in enumerate(mol_graph.edge_index.T):
        src = int(src)
        dst = int(dst)
        bond = tagged_mol.GetBondBetweenAtoms(src, dst)
        assert bond is not None

        if src == bond.GetBeginAtomIdx():
            expected_tag = int(bond.GetIntProp("endAtomPriorityTag"))
        else:
            expected_tag = int(bond.GetIntProp("beginAtomPriorityTag"))

        np.testing.assert_array_equal(mol_graph.E[edge_idx, start:stop], _encode_neighbor_tag(expected_tag))


def test_stereo_asymmetry_is_limited_to_neighbor_tag_bits():
    """Forward/reverse edge features differ only in the neighbor-tag one-hot slice."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    extra_bond_fdim = 2
    bond_features_extra = np.random.rand(mol.GetNumBonds(), extra_bond_fdim).astype(np.single)

    featurizer = StereoMoleculeMolGraphFeaturizer(extra_bond_fdim=extra_bond_fdim)
    mol_graph = featurizer(mol, bond_features_extra=bond_features_extra)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    asymmetric_found = False
    for bond in tagged_mol.GetBonds():
        bond_idx = bond.GetIdx()
        forward_row = 2 * bond_idx
        reverse_row = forward_row + 1
        begin_tag = int(bond.GetIntProp("beginAtomPriorityTag"))
        end_tag = int(bond.GetIntProp("endAtomPriorityTag"))

        if begin_tag != end_tag:
            asymmetric_found = True
            np.testing.assert_array_equal(mol_graph.E[forward_row, :start], mol_graph.E[reverse_row, :start])
            assert not np.array_equal(mol_graph.E[forward_row, start:stop], mol_graph.E[reverse_row, start:stop])
            np.testing.assert_allclose(mol_graph.E[forward_row, stop:], mol_graph.E[reverse_row, stop:])

    assert asymmetric_found


def test_stereo_featurizer_handles_molecules_with_no_bonds():
    """A single-atom molecule yields empty E and edge_index without errors."""
    mol = Chem.MolFromSmiles("[He]")
    featurizer = StereoMoleculeMolGraphFeaturizer()
    mol_graph = featurizer(mol)

    assert mol_graph.E.shape == (0, featurizer.bond_fdim)
    assert mol_graph.edge_index.shape == (2, 0)
    assert mol_graph.rev_edge_index.shape == (0,)


def test_stereo_featurizer_raises_for_invalid_extra_bond_feature_rows(mol):
    """Invalid extra bond feature row count propagates the base featurizer ValueError."""
    featurizer = StereoMoleculeMolGraphFeaturizer(extra_bond_fdim=2)
    bad_extra = np.random.rand(mol.GetNumBonds() + 1, 2).astype(np.single)

    with pytest.raises(ValueError):
        featurizer(mol, bond_features_extra=bad_extra)
