from dataclasses import InitVar, field

import numpy as np
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import VectorFeaturizer
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer

from .atom import StereoAtomFeaturizer
from .bond import StereoBondFeaturizer
from .utils import assign_neighbor_ranking


class StereoMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    r"""A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

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
    >>> mol = Chem.MolFromSmiles("C=CO")
    >>> featurizer = StereoMolGraphFeaturizer()
    >>> print(featurizer.to_string(mol))
    0: 00000100000000000000000000000000000000 0001000 000010 10000 001000 00100000 0 0.120
    1: 00000100000000000000000000000000000000 0001000 000010 10000 010000 00100000 0 0.120
    2: 00000001000000000000000000000000000000 0010000 000010 10000 010000 00100000 0 0.160
    0→1: 0 0100 1 0 1000000
    0←1: 0 0100 1 0 1000000
    0→2: 1 0000 0 0 0000000
    0←2: 1 0000 0 0 0000000
    1→2: 0 1000 1 0 1000000
    1←2: 0 1000 1 0 1000000

    """

    atom_featurizer: VectorFeaturizer[Chem.Atom] = field(default_factory=StereoAtomFeaturizer.v2)
    bond_featurizer: VectorFeaturizer[Chem.Bond] = field(default_factory=StereoBondFeaturizer)
    backward_bond_featurizer: VectorFeaturizer[Chem.Bond] | None = field(
        default_factory=StereoBondFeaturizer.backward
    )
    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        assign_neighbor_ranking(mol)
        return super().__call__(mol, atom_features_extra, bond_features_extra)
