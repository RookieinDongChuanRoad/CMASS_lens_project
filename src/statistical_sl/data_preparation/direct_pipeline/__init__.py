"""Direct source-to-canonical dataset preparation pipeline.

This package is intentionally separate from :mod:`statistical_sl.data_preparation.dataset_schema`.
The existing schema package writes canonical HDF5 files from already-prepared
observation HDF5 products.  The modules here own the newer direct path: read
simple source catalogs, attach trusted measurements, build derived numerical
blocks in memory, and only then serialize the canonical dataset.
"""

from statistical_sl.data_preparation.direct_pipeline.policies import (
    AperturePolicyRef,
    MassDefinitionPolicy,
    ProfilePolicy,
    SigmaPolicy,
    UnitPolicy,
)
from statistical_sl.data_preparation.direct_pipeline.records import (
    BaseLensRecord,
    CanonicalDatasetPayload,
    PreparedLensRecord,
    SigmaObservation,
)

__all__ = [
    "AperturePolicyRef",
    "BaseLensRecord",
    "CanonicalDatasetPayload",
    "MassDefinitionPolicy",
    "PreparedLensRecord",
    "ProfilePolicy",
    "SigmaObservation",
    "SigmaPolicy",
    "UnitPolicy",
]
