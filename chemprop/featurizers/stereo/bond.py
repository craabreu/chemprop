from enum import Enum, auto

from rdkit.Chem.rdchem import Bond, BondStereo, BondType

from chemprop.featurizers.base import (
    MultiHotFeaturizer,
    NullityFeaturizer,
    OneHotFeaturizer,
    ValueFeaturizer,
)

from .utils import get_canonical_stereo


class EdgeDirection(Enum):
    """
    Enum to specify the direction of a bond edge for stereochemical featurization.

    Attributes
    ----------
    FORWARD : Indicates the direction from the begin atom to the end atom of the bond.
    BACKWARD : Indicates the direction from the end atom to the begin atom of the bond.
    """

    FORWARD = auto()
    BACKWARD = auto()


class StereoBondFeaturizer(MultiHotFeaturizer[Bond]):
    """A vector featurizer for bond stereochemistry.

    Parameters
    ----------
    backward : bool
        Whether to use the begin atom's rank among the end atom's neighbors. If ``False``, use
        the end atom's rank among the begin atom's neighbors.

    Example
    -------
    >>> from rdkit import Chem
    >>> from chemprop.featurizers.stereo import assign_neighbor_ranking, describe_neighbor_ranking
    >>> mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")
    >>> assign_neighbor_ranking(mol)
    >>> print(describe_neighbor_ranking(mol, include_leaves=True))
    C0 C1:0
    C1 C3:0 O2:1 N6:2 C0:3 (CHI_TETRAHEDRAL_CCW)
    O2 C1:0
    C3 C1:0 C4:1
    C4 C3:0 O5:1
    O5 C4:0
    N6 C1:0
    C3-C4 STEREOTRANS
    >>> for direction in EdgeDirection:
    ...     featurizer = StereoBondFeaturizer(direction)
    ...     print(["Forward:", "Backward:"][direction == EdgeDirection.BACKWARD])
    ...     for bond in mol.GetBonds():
    ...         begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    ...         atoms = [f"{a.GetSymbol()}{a.GetIdx()}" for a in (begin, end)]
    ...         print(*atoms, featurizer.to_string(bond))
    Forward:
    C0 C1 0 1000 0 0 10000 00010
    C1 O2 0 1000 0 0 10000 10000
    C1 C3 0 1000 0 0 10000 10000
    C3 C4 0 0100 1 0 00010 10000
    C4 O5 0 1000 1 0 10000 10000
    C1 N6 0 1000 0 0 10000 10000
    Backward:
    C0 C1 0 1000 0 0 10000 10000
    C1 O2 0 1000 0 0 10000 01000
    C1 C3 0 1000 0 0 10000 10000
    C3 C4 0 0100 1 0 00010 01000
    C4 O5 0 1000 1 0 10000 01000
    C1 N6 0 1000 0 0 10000 00100

    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdBondStereo.values

    """

    def __init__(self, edge_direction: EdgeDirection):
        if edge_direction not in EdgeDirection:
            raise TypeError(f"Expected EdgeDirection, got {type(edge_direction)}")
        self.edge_direction = edge_direction
        bond_types = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC)
        stereos = (
            BondStereo.STEREONONE,
            BondStereo.STEREOANY,
            BondStereo.STEREOCIS,
            BondStereo.STEREOTRANS,
        )
        neighbor_ranks = tuple(range(4))
        rank_property = (
            "beginRankFromEnd"
            if self.edge_direction == EdgeDirection.FORWARD
            else "endRankFromBegin"
        )

        super().__init__(
            NullityFeaturizer(),
            OneHotFeaturizer(lambda b: b.GetBondType(), bond_types),
            ValueFeaturizer(lambda b: b.GetIsConjugated(), int),
            ValueFeaturizer(lambda b: b.IsInRing(), int),
            OneHotFeaturizer(get_canonical_stereo, stereos, padding=True),
            OneHotFeaturizer(lambda b: b.GetIntProp(rank_property), neighbor_ranks, padding=True),
        )

    @classmethod
    def forward(cls):
        """Return a version of this featurizer that featurizes bonds in forward direction.

        This is useful when featurizing molecules in an asymmetric way.
        """
        return cls(EdgeDirection.FORWARD)

    @classmethod
    def backward(cls):
        """Return a version of this featurizer that featurizes bonds in reverse direction.

        This is useful when featurizing molecules in an asymmetric way.
        """
        return cls(EdgeDirection.BACKWARD)
