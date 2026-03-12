import networkx as nx
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, ChiralType

from chemprop.featurizers.stereo.neighbor_tagging import (
    describe_neighbor_tagging,
    mol_with_neighbor_priority_tags,
)


def test_neighbor_tagging_chiral_center_only():
    """Annotating a single tetrahedral center yields expected chiral and neighbor tags."""
    mol = Chem.MolFromSmiles("C[C@H](O)N")

    tagged = mol_with_neighbor_priority_tags(mol)
    center = tagged.GetAtomWithIdx(1)

    assert tagged.HasProp("hasNeighborPriorityTags")
    assert tagged.GetBoolProp("hasNeighborPriorityTags")
    assert center.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW

    assert describe_neighbor_tagging(tagged) == "C1 O2:0 N3:1 C0:2 (CHI_TETRAHEDRAL_CCW)"


def test_neighbor_tagging_bond_stereo_only():
    """Annotating an alkene preserves a consistent trans bond-stereo assignment."""
    mol = Chem.MolFromSmiles("F/C=C/F")

    tagged = mol_with_neighbor_priority_tags(mol)
    bond = tagged.GetBondBetweenAtoms(1, 2)

    assert bond is not None
    assert bond.GetStereo() == BondStereo.STEREOTRANS

    for b in tagged.GetBonds():
        assert b.HasProp("beginAtomPriorityTag")
        assert b.HasProp("endAtomPriorityTag")

    assert describe_neighbor_tagging(tagged).splitlines()[-1] == "C1-C2 STEREOTRANS"


@pytest.mark.parametrize(
    "smiles,input_stereo,expected_stereo",
    [
        ("F/C=C/F", BondStereo.STEREOE, BondStereo.STEREOTRANS),
        ("F/C=C\\F", BondStereo.STEREOZ, BondStereo.STEREOCIS),
    ],
)
def test_neighbor_tagging_normalizes_ez_to_cis_trans(smiles, input_stereo, expected_stereo):
    """E/Z-marked bonds are normalized to cis/trans after neighbor tagging."""
    mol = Chem.MolFromSmiles(smiles)
    input_bond = next(bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    assert input_bond.GetStereo() == input_stereo

    tagged = mol_with_neighbor_priority_tags(mol)
    output_bond = next(
        bond for bond in tagged.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    assert output_bond.GetStereo() == expected_stereo
    assert output_bond.GetStereo() not in {BondStereo.STEREOE, BondStereo.STEREOZ}


def test_neighbor_tagging_combined_atom_and_bond_stereo():
    """A molecule with atom and bond stereo receives both tag types consistently."""
    mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")

    tagged = mol_with_neighbor_priority_tags(mol)
    center = tagged.GetAtomWithIdx(1)
    bond = tagged.GetBondBetweenAtoms(3, 4)

    assert center.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW
    assert bond is not None
    assert bond.GetStereo() == BondStereo.STEREOTRANS

    desc = describe_neighbor_tagging(tagged)
    assert "C1 C3:0 O2:1 N6:2 C0:3 (CHI_TETRAHEDRAL_CCW)" in desc
    assert "C3-C4 STEREOTRANS" in desc


def test_neighbor_tagging_idempotent():
    """Running neighbor tagging twice does not change the tagged molecule."""
    mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")

    once = mol_with_neighbor_priority_tags(mol)
    twice = mol_with_neighbor_priority_tags(once)

    assert twice.HasProp("hasNeighborPriorityTags")
    assert twice.GetBoolProp("hasNeighborPriorityTags")

    assert Chem.MolToSmiles(once, isomericSmiles=True) == Chem.MolToSmiles(
        twice, isomericSmiles=True
    )
    assert describe_neighbor_tagging(once) == describe_neighbor_tagging(twice)


def test_neighbor_tagging_symmetry_tie_case():
    """Symmetric substituents produce tied neighbor priority tags at the center atom."""
    # In this symmetric case, the central carbon's neighbor tags are tied.
    mol = Chem.MolFromSmiles("CC(C)(C)C")
    tagged = mol_with_neighbor_priority_tags(mol)

    desc = describe_neighbor_tagging(tagged, include_leaves=True)
    center_line = next(line for line in desc.splitlines() if line.startswith("C1 "))

    assert center_line == "C1 C0:0 C2:0 C3:0 C4:0"


def _to_tagged_nx_graph(mol):
    graph = nx.Graph()

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        neighbor_tags = []
        for bond in atom.GetBonds():
            if bond.GetBeginAtomIdx() == atom_idx:
                neighbor_tags.append(int(bond.GetIntProp("endAtomPriorityTag")))
            else:
                neighbor_tags.append(int(bond.GetIntProp("beginAtomPriorityTag")))

        graph.add_node(
            atom_idx,
            atomic_num=atom.GetAtomicNum(),
            chiral_tag=atom.GetChiralTag().name,
            neighbor_tags=tuple(sorted(neighbor_tags)),
        )

    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        begin_tag = int(bond.GetIntProp("beginAtomPriorityTag"))
        end_tag = int(bond.GetIntProp("endAtomPriorityTag"))
        graph.add_edge(
            begin_idx,
            end_idx,
            bond_type=str(bond.GetBondType()),
            stereo=bond.GetStereo().name,
            endpoint_tags=tuple(sorted((begin_tag, end_tag))),
        )

    return graph


@pytest.mark.parametrize("base_smiles", ["C[C@H](O)N", "F/C=C/F", "C[C@](O)(/C=C/O)N"])
def test_neighbor_tagging_invariant_to_input_smiles_atom_order(base_smiles):
    """Tagging results are invariant under equivalent SMILES atom-order permutations."""
    base_mol = Chem.MolFromSmiles(base_smiles)
    smiles_variants = {
        Chem.MolToSmiles(base_mol, canonical=False, isomericSmiles=True, rootedAtAtom=i)
        for i in range(base_mol.GetNumAtoms())
    }

    assert len(smiles_variants) > 1

    tagged_graphs = [
        _to_tagged_nx_graph(mol_with_neighbor_priority_tags(Chem.MolFromSmiles(smi)))
        for smi in smiles_variants
    ]
    reference = tagged_graphs[0]

    for graph in tagged_graphs[1:]:
        assert nx.is_isomorphic(
            reference, graph, node_match=lambda a, b: a == b, edge_match=lambda a, b: a == b
        )


def test_heavy_neighbor_tags_match_for_implicit_vs_explicit_hydrogen():
    """Heavy-atom neighbor tags match whether the chiral hydrogen is implicit or explicit."""
    implicit_h = Chem.MolFromSmiles("C[C@H](O)N")
    explicit_h = Chem.MolFromSmiles("[H][C@](C)(O)N")

    tagged_implicit = mol_with_neighbor_priority_tags(implicit_h)
    tagged_explicit = mol_with_neighbor_priority_tags(explicit_h)

    implicit_center = next(
        atom
        for atom in tagged_implicit.GetAtoms()
        if atom.GetAtomicNum() != 1 and atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    )
    explicit_center = next(
        atom
        for atom in tagged_explicit.GetAtoms()
        if atom.GetAtomicNum() != 1 and atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
    )

    implicit_heavy_neighbor_tags = {}
    for bond in implicit_center.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin_idx == implicit_center.GetIdx():
            neighbor = tagged_implicit.GetAtomWithIdx(end_idx)
            tag = int(bond.GetIntProp("endAtomPriorityTag"))
        else:
            neighbor = tagged_implicit.GetAtomWithIdx(begin_idx)
            tag = int(bond.GetIntProp("beginAtomPriorityTag"))
        if neighbor.GetAtomicNum() != 1:
            implicit_heavy_neighbor_tags[neighbor.GetAtomicNum()] = tag

    explicit_heavy_neighbor_tags = {}
    for bond in explicit_center.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin_idx == explicit_center.GetIdx():
            neighbor = tagged_explicit.GetAtomWithIdx(end_idx)
            tag = int(bond.GetIntProp("endAtomPriorityTag"))
        else:
            neighbor = tagged_explicit.GetAtomWithIdx(begin_idx)
            tag = int(bond.GetIntProp("beginAtomPriorityTag"))
        if neighbor.GetAtomicNum() != 1:
            explicit_heavy_neighbor_tags[neighbor.GetAtomicNum()] = tag

    assert implicit_heavy_neighbor_tags == explicit_heavy_neighbor_tags
