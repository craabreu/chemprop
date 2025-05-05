from typing import Sequence

from rdkit.Chem.rdchem import ChiralType, HybridizationType

from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.base import OneHotFeaturizer, ValueFeaturizer

from .utils import get_canonical_chiral_tag


class StereoAtomFeaturizer(MultiHotAtomFeaturizer):
    """A subclass of :class:`MultiHotAtomFeaturizer` that uses canonical chiral tags.

    Example
    -------
    >>> from rdkit import Chem
    >>> from chemprop.featurizers.stereo import assign_neighbor_ranking, describe_neighbor_ranking
    >>> for smiles in ["C[C@H](O)N", "C[C@@H](N)O"]:
    ...     print("Molecule:", smiles)
    ...     mol = Chem.MolFromSmiles(smiles)
    ...     assign_neighbor_ranking(mol)
    ...     print(describe_neighbor_ranking(mol))
    ...     featurizer = StereoAtomFeaturizer.v2()
    ...     for atom in mol.GetAtoms():
    ...         print(featurizer.to_string(atom))
    Molecule: C[C@H](O)N
    C1 O2:0 N3:1 C0:2 (CHI_TETRAHEDRAL_CCW)
    00000100000000000000000000000000000000 0000100 000010 10000 000100 00001000 0 0.120
    00000100000000000000000000000000000000 0000100 000010 00100 010000 00001000 0 0.120
    00000001000000000000000000000000000000 0010000 000010 10000 010000 00001000 0 0.160
    00000010000000000000000000000000000000 0001000 000010 10000 001000 00001000 0 0.140
    Molecule: C[C@@H](N)O
    C1 O3:0 N2:1 C0:2 (CHI_TETRAHEDRAL_CCW)
    00000100000000000000000000000000000000 0000100 000010 10000 000100 00001000 0 0.120
    00000100000000000000000000000000000000 0000100 000010 00100 010000 00001000 0 0.120
    00000010000000000000000000000000000000 0001000 000010 10000 001000 00001000 0 0.140
    00000001000000000000000000000000000000 0010000 000010 10000 010000 00001000 0 0.160

    """

    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[ChiralType | int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[HybridizationType | int],
    ):
        self.atomic_nums = atomic_nums
        self.degrees = degrees
        self.formal_charges = formal_charges
        self.chiral_tags = chiral_tags
        self.num_Hs = num_Hs
        self.hybridizations = hybridizations

        super(MultiHotAtomFeaturizer, self).__init__(
            OneHotFeaturizer(lambda a: a.GetAtomicNum(), atomic_nums, padding=True),
            OneHotFeaturizer(lambda a: a.GetTotalDegree(), degrees, padding=True),
            OneHotFeaturizer(lambda a: a.GetFormalCharge(), formal_charges, padding=True),
            OneHotFeaturizer(get_canonical_chiral_tag, chiral_tags, padding=True),
            OneHotFeaturizer(lambda a: a.GetTotalNumHs(), num_Hs, padding=True),
            OneHotFeaturizer(lambda a: a.GetHybridization(), hybridizations, padding=True),
            ValueFeaturizer(lambda a: a.GetIsAromatic(), int),
            ValueFeaturizer(lambda a: 0.01 * a.GetMass(), float),
        )
