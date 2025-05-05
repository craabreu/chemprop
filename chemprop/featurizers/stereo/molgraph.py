from dataclasses import InitVar, dataclass, field

import numpy as np
from rdkit.Chem.rdchem import Atom, Bond, Mol

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import VectorFeaturizer
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer

from .atom import StereoAtomFeaturizer
from .bond import StereoBondFeaturizer
from .utils import assign_neighbor_ranking


@dataclass
class StereoMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    """A :class:`MolGraphFeaturizer` that computes enhanced stereochemical features.

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    backward_bond_featurizer : BondFeaturizer | None, default=None
        the featurizer with which to compute feature representations for backward bonds in a
        molecule. If this is ``None``, the ``bond_featurizer`` will be used for both forward and
        backward bonds. A forward bond is defined as starting at ``bond.GetBeginAtom()`` and ending
        at ``bond.GetEndAtom()``, while a reversed bond starts at ``bond.GetEndAtom()`` and ends at
        ``bond.GetBeginAtom()``.
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond

    Example
    -------
    >>> from rdkit import Chem
    >>> from chemprop.featurizers.stereo import describe_neighbor_ranking
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> featurizer = StereoMolGraphFeaturizer()
    >>> print(featurizer.to_string(mol))
    0: 00000100000000000000000000000000000000 0000100 000010 10000 000100 00001000 0 0.120
    1: 00000100000000000000000000000000000000 0000100 000010 01000 010000 00001000 0 0.120
    2: 00000010000000000000000000000000000000 0001000 000010 10000 001000 00001000 0 0.140
    3: 00000001000000000000000000000000000000 0010000 000010 10000 010000 00001000 0 0.160
    0→1: 0 1000 0 0 10000 00100
    0←1: 0 1000 0 0 10000 10000
    1→2: 0 1000 0 0 10000 10000
    1←2: 0 1000 0 0 10000 01000
    1→3: 0 1000 0 0 10000 10000
    1←3: 0 1000 0 0 10000 10000
    >>> print(describe_neighbor_ranking(mol, include_leaves=True))
    C0 C1:0
    C1 O3:0 N2:1 C0:2 (CHI_TETRAHEDRAL_CW)
    N2 C1:0
    O3 C1:0

    """

    atom_featurizer: VectorFeaturizer[Atom] = field(default_factory=StereoAtomFeaturizer.v2)
    bond_featurizer: VectorFeaturizer[Bond] = field(default_factory=StereoBondFeaturizer.forward)
    backward_bond_featurizer: VectorFeaturizer[Bond] = field(
        default_factory=StereoBondFeaturizer.backward
    )
    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0

    def __call__(
        self,
        mol: Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        assign_neighbor_ranking(mol)
        return super().__call__(mol, atom_features_extra, bond_features_extra)

    def to_string(self, mol: Mol, decimals: int = 3) -> str:
        assign_neighbor_ranking(mol)
        return super().to_string(mol, decimals)
