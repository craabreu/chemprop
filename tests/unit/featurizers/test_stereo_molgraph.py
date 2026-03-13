import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.stereo.molgraph import (
    CHIRAL_CENTER_TAGS,
    NUM_NEIGHBOR_TAG_BITS,
    STEREOGENIC_BOND_TAGS,
    StereoMolGraphFeaturizer,
)
from chemprop.featurizers.stereo.neighbor_tagging import (
    mol_with_neighbor_priority_tags,
    normalize_chiral_tags_to_ccw,
)


def _encode_neighbor_tag(tag):
    one_hot = np.zeros(NUM_NEIGHBOR_TAG_BITS, dtype=np.single)
    one_hot[min(tag, 3)] = 1.0
    return one_hot


def _expected_tag_for_directed_edge(bond, source_atom_idx):
    if source_atom_idx == bond.GetBeginAtomIdx():
        return int(bond.GetIntProp("endAtomPriorityTag"))

    return int(bond.GetIntProp("beginAtomPriorityTag"))


def _stereo_target_atoms(mol):
    targets = {
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetChiralTag() in CHIRAL_CENTER_TAGS
    }
    for bond in mol.GetBonds():
        if bond.GetStereo() in STEREOGENIC_BOND_TAGS:
            targets.add(bond.GetBeginAtomIdx())
            targets.add(bond.GetEndAtomIdx())

    return targets


def _assert_same_molgraph(actual, expected):
    np.testing.assert_array_equal(actual.V, expected.V)
    np.testing.assert_array_equal(actual.E, expected.E)
    np.testing.assert_array_equal(actual.edge_index, expected.edge_index)
    np.testing.assert_array_equal(actual.rev_edge_index, expected.rev_edge_index)


def test_stereo_featurizer_constructor_supports_options():
    """Constructor accepts stereo options and uses expected defaults."""
    default_featurizer = StereoMolGraphFeaturizer()
    assert default_featurizer.stereo_atoms_only
    assert default_featurizer.normalize_chiral_tags

    full_featurizer = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=True
    )
    assert not full_featurizer.stereo_atoms_only
    assert full_featurizer.normalize_chiral_tags


def test_normalize_chiral_tags_option_matches_manual_normalization():
    """normalize_chiral_tags=True matches explicit pre-normalization before featurization."""
    source = Chem.MolFromSmiles("C[C@H](O)N")
    tagged = mol_with_neighbor_priority_tags(source)
    center = next(
        atom for atom in tagged.GetAtoms() if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    )
    center.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)

    auto_featurizer = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=True
    )
    auto_graph = auto_featurizer(tagged)

    expected = Chem.Mol(tagged)
    normalize_chiral_tags_to_ccw(expected)
    manual_featurizer = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=False
    )
    manual_graph = manual_featurizer(expected)

    _assert_same_molgraph(auto_graph, manual_graph)


def test_normalize_chiral_tags_option_matches_manual_normalization_stereo_atoms_only():
    """normalize_chiral_tags interacts correctly with stereo_atoms_only=True."""
    source = Chem.MolFromSmiles("C[C@H](O)N")
    tagged = mol_with_neighbor_priority_tags(source)
    center = next(
        atom for atom in tagged.GetAtoms() if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    )
    center.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)

    auto_featurizer = StereoMolGraphFeaturizer(
        stereo_atoms_only=True, normalize_chiral_tags=True
    )
    auto_graph = auto_featurizer(tagged)

    expected = Chem.Mol(tagged)
    normalize_chiral_tags_to_ccw(expected)
    manual_featurizer = StereoMolGraphFeaturizer(
        stereo_atoms_only=True, normalize_chiral_tags=False
    )
    manual_graph = manual_featurizer(expected)

    _assert_same_molgraph(auto_graph, manual_graph)


def test_normalize_chiral_tags_option_changes_features_for_cw_tagged_input():
    """Enabling normalization changes graph features when a CW-tagged center is present."""
    source = Chem.MolFromSmiles("C[C@H](O)N")
    tagged = mol_with_neighbor_priority_tags(source)
    center = next(
        atom for atom in tagged.GetAtoms() if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    )
    center.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)

    no_norm_graph = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=False
    )(tagged)
    norm_graph = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=True
    )(tagged)

    assert not np.array_equal(no_norm_graph.V, norm_graph.V) or not np.array_equal(
        no_norm_graph.E, norm_graph.E
    )


def test_normalize_chiral_tags_option_is_noop_when_centers_are_already_ccw():
    """Enabling chiral-tag normalization does not change already normalized molecules."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    no_norm_graph = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=False
    )(mol)
    norm_graph = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=True
    )(mol)

    _assert_same_molgraph(no_norm_graph, norm_graph)


def test_normalize_chiral_tags_option_is_noop_on_non_chiral_stereo_bond_case():
    """Normalization does not affect molecules with bond stereo but no chiral centers."""
    mol = Chem.MolFromSmiles("F/C=C/F")
    no_norm_graph = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=False
    )(mol)
    norm_graph = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=True
    )(mol)

    _assert_same_molgraph(no_norm_graph, norm_graph)


def test_normalize_chiral_tags_option_does_not_mutate_pretagged_input_molecule():
    """normalize_chiral_tags=True leaves a pre-tagged input molecule unchanged."""
    source = Chem.MolFromSmiles("C[C@H](O)N")
    tagged = mol_with_neighbor_priority_tags(source)
    center = next(
        atom for atom in tagged.GetAtoms() if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    )
    center.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)
    before_tag = center.GetChiralTag()
    before_begin_tags = [int(bond.GetIntProp("beginAtomPriorityTag")) for bond in tagged.GetBonds()]
    before_end_tags = [int(bond.GetIntProp("endAtomPriorityTag")) for bond in tagged.GetBonds()]

    featurizer = StereoMolGraphFeaturizer(
        stereo_atoms_only=False, normalize_chiral_tags=True
    )
    _ = featurizer(tagged)

    assert center.GetChiralTag() == before_tag
    assert [
        int(bond.GetIntProp("beginAtomPriorityTag")) for bond in tagged.GetBonds()
    ] == before_begin_tags
    assert [
        int(bond.GetIntProp("endAtomPriorityTag")) for bond in tagged.GetBonds()
    ] == before_end_tags


def test_stereo_bond_extra_features_are_appended_by_feature_dimension(mol):
    """Neighbor-tag one-hots and user bond extras are both appended to E."""
    n_bonds = mol.GetNumBonds()
    extra_bond_fdim = 3
    bond_features_extra = np.random.rand(n_bonds, extra_bond_fdim).astype(np.single)

    featurizer = StereoMolGraphFeaturizer(
        extra_bond_fdim=extra_bond_fdim, stereo_atoms_only=False, normalize_chiral_tags=False
    )
    mol_graph = featurizer(mol, bond_features_extra=bond_features_extra)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    for bond in tagged_mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin_tag = int(bond.GetIntProp("beginAtomPriorityTag"))
        end_tag = int(bond.GetIntProp("endAtomPriorityTag"))
        np.testing.assert_array_equal(
            mol_graph.E[2 * bond_idx, start:stop], _encode_neighbor_tag(end_tag)
        )
        np.testing.assert_array_equal(
            mol_graph.E[2 * bond_idx + 1, start:stop], _encode_neighbor_tag(begin_tag)
        )

    np.testing.assert_allclose(mol_graph.E[::2, stop:], bond_features_extra)


def test_stereo_neighbor_tag_bits_can_be_asymmetric_between_edge_directions():
    """Directed edge features can differ when endpoint neighbor tags are different."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    featurizer = StereoMolGraphFeaturizer(stereo_atoms_only=False)
    mol_graph = featurizer(mol)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    asymmetric_found = False
    for bond in tagged_mol.GetBonds():
        bond_idx = bond.GetIdx()
        if int(bond.GetIntProp("beginAtomPriorityTag")) != int(
            bond.GetIntProp("endAtomPriorityTag")
        ):
            asymmetric_found = True
            forward = mol_graph.E[2 * bond_idx, start:stop]
            reverse = mol_graph.E[2 * bond_idx + 1, start:stop]
            assert not np.array_equal(forward, reverse)

    assert asymmetric_found


def test_stereo_neighbor_tag_bucket_for_tag_three_maps_to_last_bit():
    """Neighbor tag 3 is encoded in the last bucket bit (0001)."""
    mol = Chem.MolFromSmiles("C[C@](F)(Cl)Br")
    featurizer = StereoMolGraphFeaturizer(stereo_atoms_only=False)
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
    featurizer = StereoMolGraphFeaturizer(stereo_atoms_only=False)
    mol_graph = featurizer(mol)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    for edge_idx, (src, dst) in enumerate(mol_graph.edge_index.T):
        src = int(src)
        dst = int(dst)
        bond = tagged_mol.GetBondBetweenAtoms(src, dst)
        assert bond is not None

        expected_tag = _expected_tag_for_directed_edge(bond, src)

        np.testing.assert_array_equal(
            mol_graph.E[edge_idx, start:stop], _encode_neighbor_tag(expected_tag)
        )


def test_stereo_asymmetry_is_limited_to_neighbor_tag_bits():
    """Forward/reverse edge features differ only in the neighbor-tag one-hot slice."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    extra_bond_fdim = 2
    bond_features_extra = np.random.rand(mol.GetNumBonds(), extra_bond_fdim).astype(np.single)

    featurizer = StereoMolGraphFeaturizer(
        extra_bond_fdim=extra_bond_fdim, stereo_atoms_only=False
    )
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
            np.testing.assert_array_equal(
                mol_graph.E[forward_row, :start], mol_graph.E[reverse_row, :start]
            )
            assert not np.array_equal(
                mol_graph.E[forward_row, start:stop], mol_graph.E[reverse_row, start:stop]
            )
            np.testing.assert_allclose(
                mol_graph.E[forward_row, stop:], mol_graph.E[reverse_row, stop:]
            )

    assert asymmetric_found


@pytest.mark.parametrize("smiles", ["C[C@H](O)N", "F/C=C/F"])
def test_stereo_atoms_only_default_encodes_edges_into_stereo_relevant_atoms(smiles):
    """Default mode encodes one-hots only for directed edges into stereo-relevant atoms."""
    mol = Chem.MolFromSmiles(smiles)
    featurizer = StereoMolGraphFeaturizer()
    mol_graph = featurizer(mol)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS
    targets = _stereo_target_atoms(tagged_mol)
    assert targets

    saw_target_edge = False
    saw_non_target_edge = False
    zero_bits = np.zeros(NUM_NEIGHBOR_TAG_BITS, dtype=np.single)
    for edge_idx, (src, dst) in enumerate(mol_graph.edge_index.T):
        src = int(src)
        dst = int(dst)
        bond = tagged_mol.GetBondBetweenAtoms(src, dst)
        assert bond is not None

        if dst in targets:
            saw_target_edge = True
            expected_tag = _expected_tag_for_directed_edge(bond, src)
            np.testing.assert_array_equal(
                mol_graph.E[edge_idx, start:stop], _encode_neighbor_tag(expected_tag)
            )
        else:
            saw_non_target_edge = True
            np.testing.assert_array_equal(mol_graph.E[edge_idx, start:stop], zero_bits)

    assert saw_target_edge
    assert saw_non_target_edge


def test_stereo_atoms_only_encodes_only_edges_converging_to_chiral_center():
    """For chiral-center-only stereo, only incoming directed edges to the center are encoded."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    featurizer = StereoMolGraphFeaturizer(stereo_atoms_only=True)
    mol_graph = featurizer(mol)
    tagged_mol = mol_with_neighbor_priority_tags(mol)

    center = next(
        atom for atom in tagged_mol.GetAtoms() if atom.GetChiralTag() in CHIRAL_CENTER_TAGS
    )
    center_idx = center.GetIdx()

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS
    zero_bits = np.zeros(NUM_NEIGHBOR_TAG_BITS, dtype=np.single)

    incoming_count = 0
    outgoing_count = 0
    for edge_idx, (src, dst) in enumerate(mol_graph.edge_index.T):
        src = int(src)
        dst = int(dst)
        bond = tagged_mol.GetBondBetweenAtoms(src, dst)
        assert bond is not None

        if dst == center_idx:
            incoming_count += 1
            expected_tag = _expected_tag_for_directed_edge(bond, src)
            np.testing.assert_array_equal(
                mol_graph.E[edge_idx, start:stop], _encode_neighbor_tag(expected_tag)
            )
        elif src == center_idx:
            outgoing_count += 1
            np.testing.assert_array_equal(mol_graph.E[edge_idx, start:stop], zero_bits)

    assert incoming_count == center.GetDegree()
    assert outgoing_count == center.GetDegree()


def test_stereo_atoms_only_default_keeps_neighbor_bits_zero_without_stereo_atoms():
    """Default mode leaves neighbor one-hot bits zero when no stereo-relevant atoms exist."""
    mol = Chem.MolFromSmiles("CCC")
    featurizer = StereoMolGraphFeaturizer()
    mol_graph = featurizer(mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS
    np.testing.assert_array_equal(mol_graph.E[:, start:stop], 0.0)


def test_stereo_atoms_only_option_changes_behavior_on_non_stereogenic_molecule():
    """stereo_atoms_only toggles neighbor-tag encoding on non-stereogenic molecules."""
    mol = Chem.MolFromSmiles("CCC")
    featurizer_stereo_only = StereoMolGraphFeaturizer(stereo_atoms_only=True)
    featurizer_all_atoms = StereoMolGraphFeaturizer(stereo_atoms_only=False)
    graph_stereo_only = featurizer_stereo_only(mol)
    graph_all_atoms = featurizer_all_atoms(mol)

    start = len(featurizer_stereo_only.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS
    np.testing.assert_array_equal(graph_stereo_only.E[:, start:stop], 0.0)

    assert np.any(graph_all_atoms.E[:, start:stop] != 0.0)
    np.testing.assert_array_equal(graph_all_atoms.E[:, start:stop].sum(axis=1), 1.0)


def test_stereo_and_simple_match_on_non_stereo_molecule_except_neighbor_tag_bits():
    """On non-stereo molecules, Stereo and Simple graphs match except inserted tag-bit columns."""
    mol = Chem.MolFromSmiles("CCCO")
    simple_featurizer = SimpleMoleculeMolGraphFeaturizer()
    stereo_featurizer = StereoMolGraphFeaturizer()

    simple_graph = simple_featurizer(mol)
    stereo_graph = stereo_featurizer(mol)

    start = len(stereo_featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    np.testing.assert_array_equal(stereo_graph.V, simple_graph.V)
    np.testing.assert_array_equal(stereo_graph.edge_index, simple_graph.edge_index)
    np.testing.assert_array_equal(stereo_graph.rev_edge_index, simple_graph.rev_edge_index)
    np.testing.assert_array_equal(stereo_graph.E[:, :start], simple_graph.E)
    np.testing.assert_array_equal(stereo_graph.E[:, start:stop], 0.0)


def test_stereo_and_simple_match_on_non_stereo_molecule_with_extra_bond_features():
    """With extra bond features, Stereo and Simple differ only by zeroed inserted tag bits."""
    mol = Chem.MolFromSmiles("CCCO")
    extra_bond_fdim = 3
    bond_features_extra = np.random.rand(mol.GetNumBonds(), extra_bond_fdim).astype(np.single)

    simple_featurizer = SimpleMoleculeMolGraphFeaturizer(extra_bond_fdim=extra_bond_fdim)
    stereo_featurizer = StereoMolGraphFeaturizer(extra_bond_fdim=extra_bond_fdim)

    simple_graph = simple_featurizer(mol, bond_features_extra=bond_features_extra)
    stereo_graph = stereo_featurizer(mol, bond_features_extra=bond_features_extra)

    start = len(stereo_featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS

    np.testing.assert_array_equal(stereo_graph.V, simple_graph.V)
    np.testing.assert_array_equal(stereo_graph.edge_index, simple_graph.edge_index)
    np.testing.assert_array_equal(stereo_graph.rev_edge_index, simple_graph.rev_edge_index)
    np.testing.assert_array_equal(stereo_graph.E[:, :start], simple_graph.E[:, :start])
    np.testing.assert_array_equal(stereo_graph.E[:, start:stop], 0.0)
    np.testing.assert_array_equal(stereo_graph.E[:, stop:], simple_graph.E[:, start:])


def test_stereo_atoms_only_default_matches_explicit_true():
    """Default and explicit stereo_atoms_only=True produce identical outputs."""
    mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")
    default_featurizer = StereoMolGraphFeaturizer()
    explicit_featurizer = StereoMolGraphFeaturizer(stereo_atoms_only=True)

    default_graph = default_featurizer(mol)
    explicit_graph = explicit_featurizer(mol)

    np.testing.assert_array_equal(default_graph.E, explicit_graph.E)
    np.testing.assert_array_equal(default_graph.V, explicit_graph.V)
    np.testing.assert_array_equal(default_graph.edge_index, explicit_graph.edge_index)
    np.testing.assert_array_equal(default_graph.rev_edge_index, explicit_graph.rev_edge_index)


def test_stereo_atoms_only_with_extra_bond_features_preserves_extra_columns():
    """Default mode keeps non-target tag bits zero while preserving extra bond features."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    extra_bond_fdim = 2
    bond_features_extra = np.random.rand(mol.GetNumBonds(), extra_bond_fdim).astype(np.single)

    featurizer = StereoMolGraphFeaturizer(
        extra_bond_fdim=extra_bond_fdim, stereo_atoms_only=True
    )
    mol_graph = featurizer(mol, bond_features_extra=bond_features_extra)
    tagged_mol = mol_with_neighbor_priority_tags(mol)
    targets = _stereo_target_atoms(tagged_mol)

    start = len(featurizer.bond_featurizer)
    stop = start + NUM_NEIGHBOR_TAG_BITS
    zero_bits = np.zeros(NUM_NEIGHBOR_TAG_BITS, dtype=np.single)

    for edge_idx, (src, dst) in enumerate(mol_graph.edge_index.T):
        src = int(src)
        dst = int(dst)
        bond = tagged_mol.GetBondBetweenAtoms(src, dst)
        assert bond is not None

        if dst in targets:
            expected_tag = _expected_tag_for_directed_edge(bond, src)
            np.testing.assert_array_equal(
                mol_graph.E[edge_idx, start:stop], _encode_neighbor_tag(expected_tag)
            )
        else:
            np.testing.assert_array_equal(mol_graph.E[edge_idx, start:stop], zero_bits)

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        np.testing.assert_allclose(mol_graph.E[2 * bond_idx, stop:], bond_features_extra[bond_idx])
        np.testing.assert_allclose(
            mol_graph.E[2 * bond_idx + 1, stop:], bond_features_extra[bond_idx]
        )


def test_stereo_featurizer_does_not_mutate_input_molecule():
    """Featurization leaves the input molecule's stereo and properties unchanged."""
    mol = Chem.MolFromSmiles("F/C=C/F")
    before_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    before_stereo = mol.GetBondBetweenAtoms(1, 2).GetStereo()

    featurizer = StereoMolGraphFeaturizer()
    _ = featurizer(mol)

    assert Chem.MolToSmiles(mol, isomericSmiles=True) == before_smi
    assert mol.GetBondBetweenAtoms(1, 2).GetStereo() == before_stereo
    assert not mol.HasProp("hasNeighborPriorityTags")
    assert all(not bond.HasProp("beginAtomPriorityTag") for bond in mol.GetBonds())
    assert all(not bond.HasProp("endAtomPriorityTag") for bond in mol.GetBonds())


def test_stereo_featurizer_repeated_calls_are_deterministic():
    """Calling the featurizer repeatedly on the same input yields identical graphs."""
    mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")
    featurizer = StereoMolGraphFeaturizer(stereo_atoms_only=False)

    graph_a = featurizer(mol)
    graph_b = featurizer(mol)

    np.testing.assert_array_equal(graph_a.V, graph_b.V)
    np.testing.assert_array_equal(graph_a.E, graph_b.E)
    np.testing.assert_array_equal(graph_a.edge_index, graph_b.edge_index)
    np.testing.assert_array_equal(graph_a.rev_edge_index, graph_b.rev_edge_index)


def test_stereo_featurizer_handles_molecules_with_no_bonds():
    """A single-atom molecule yields empty E and edge_index without errors."""
    mol = Chem.MolFromSmiles("[He]")
    featurizer = StereoMolGraphFeaturizer()
    mol_graph = featurizer(mol)

    assert mol_graph.E.shape == (0, featurizer.bond_fdim)
    assert mol_graph.edge_index.shape == (2, 0)
    assert mol_graph.rev_edge_index.shape == (0,)


def test_stereo_featurizer_raises_for_invalid_extra_bond_feature_rows(mol):
    """Invalid extra bond feature row count propagates the base featurizer ValueError."""
    featurizer = StereoMolGraphFeaturizer(extra_bond_fdim=2)
    bad_extra = np.random.rand(mol.GetNumBonds() + 1, 2).astype(np.single)

    with pytest.raises(ValueError):
        featurizer(mol, bond_features_extra=bad_extra)
