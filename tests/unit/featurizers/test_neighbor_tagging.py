from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, ChiralType

from chemprop.featurizers.stereo.neighbor_tagging import (
    describe_neighbor_tagging,
    mol_with_neighbor_priority_tags,
)


def test_neighbor_tagging_chiral_center_only():
    mol = Chem.MolFromSmiles("C[C@H](O)N")

    tagged = mol_with_neighbor_priority_tags(mol)
    center = tagged.GetAtomWithIdx(1)

    assert tagged.HasProp("hasNeighborPriorityTags")
    assert tagged.GetBoolProp("hasNeighborPriorityTags")
    assert center.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW

    assert (
        describe_neighbor_tagging(tagged)
        == "C1 O2:0 N3:1 C0:2 (CHI_TETRAHEDRAL_CCW)"
    )


def test_neighbor_tagging_bond_stereo_only():
    mol = Chem.MolFromSmiles("F/C=C/F")

    tagged = mol_with_neighbor_priority_tags(mol)
    bond = tagged.GetBondBetweenAtoms(1, 2)

    assert bond is not None
    assert bond.GetStereo() == BondStereo.STEREOTRANS

    for b in tagged.GetBonds():
        assert b.HasProp("beginAtomPriorityTag")
        assert b.HasProp("endAtomPriorityTag")

    assert describe_neighbor_tagging(tagged).splitlines()[-1] == "C1-C2 STEREOTRANS"


def test_neighbor_tagging_combined_atom_and_bond_stereo():
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
    # In this symmetric case, the central carbon's neighbor tags are tied.
    mol = Chem.MolFromSmiles("CC(C)(C)C")
    tagged = mol_with_neighbor_priority_tags(mol)

    desc = describe_neighbor_tagging(tagged, include_leaves=True)
    center_line = next(line for line in desc.splitlines() if line.startswith("C1 "))

    assert center_line == "C1 C0:0 C2:0 C3:0 C4:0"
