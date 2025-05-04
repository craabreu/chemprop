from rdkit import Chem

from chemprop.featurizers.base import (
    MultiHotFeaturizer,
    NullityFeaturizer,
    OneHotFeaturizer,
    ValueFeaturizer,
)

from .utils import get_canonical_stereo


class StereoBondFeaturizer(MultiHotFeaturizer[Chem.Bond]):
    """A vector featurizer for bond stereochemistry.

    Parameters
    ----------
    backward : bool
        Whether to use the begin atom's rank among the end atom's neighbors. The default is
        ``False``, i.e., use the end atom's rank among the begin atom's neighbors.

    Example
    -------
    >>> from rdkit import Chem
    >>> from chemprop.featurizers.stereo import assign_neighbor_ranking, neighbor_ranking_string
    >>> mol = Chem.MolFromSmiles("C[C@](O)(/C=C/O)N")
    >>> assign_neighbor_ranking(mol)
    >>> for atom in mol.GetAtoms():
    ...     print(neighbor_ranking_string(atom))
    C0 C1:0
    C1 C3:0 O2:1 N6:2 C0:3
    O2 C1:0
    C3 C1:0 C4:1
    C4 C3:0 O5:1
    O5 C4:0
    N6 C1:0
    >>> for backward in [False, True]:
    ...     featurizer = StereoBondFeaturizer(backward=backward)
    ...     print(["Forward:", "Backward:"][backward])
    ...     for bond in mol.GetBonds():
    ...         begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    ...         atoms = [f"{a.GetSymbol()}{a.GetIdx()}" for a in (begin, end)]
    ...         print(*atoms, featurizer.to_string(bond))
    Forward:
    C0 C1 0 1000 0 0 10000 10000
    C1 O2 0 1000 0 0 10000 01000
    C1 C3 0 1000 0 0 10000 10000
    C3 C4 0 0100 1 0 00010 01000
    C4 O5 0 1000 1 0 10000 01000
    C1 N6 0 1000 0 0 10000 00100
    Backward:
    C0 C1 0 1000 0 0 10000 00010
    C1 O2 0 1000 0 0 10000 10000
    C1 C3 0 1000 0 0 10000 10000
    C3 C4 0 0100 1 0 00010 10000
    C4 O5 0 1000 1 0 10000 10000
    C1 N6 0 1000 0 0 10000 10000

    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values

    """

    def __init__(self, backward: bool = False):
        bond_types = (
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        )
        stereos = (
            Chem.BondStereo.STEREONONE,
            Chem.BondStereo.STEREOANY,
            Chem.BondStereo.STEREOCIS,
            Chem.BondStereo.STEREOTRANS,
        )
        neighbor_ranks = tuple(range(4))
        rank_property = "beginRankFromEnd" if backward else "endRankFromBegin"

        super().__init__(
            NullityFeaturizer(),
            OneHotFeaturizer(lambda b: b.GetBondType(), bond_types),
            ValueFeaturizer(lambda b: b.GetIsConjugated(), int),
            ValueFeaturizer(lambda b: b.IsInRing(), int),
            OneHotFeaturizer(get_canonical_stereo, stereos, padding=True),
            OneHotFeaturizer(lambda b: b.GetIntProp(rank_property), neighbor_ranks, padding=True),
        )
