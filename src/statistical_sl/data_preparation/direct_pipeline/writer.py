"""HDF5 writer for direct canonical dataset payloads."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from statistical_sl.data_preparation.direct_pipeline.records import CanonicalDatasetPayload
from statistical_sl.data_preparation.direct_pipeline.validator import (
    validate_canonical_dataset_payload,
    validate_canonical_hdf5,
)
from statistical_sl.core.canonical_schema import (
    BLOCK_LENSES,
    BLOCK_LENSING_CROSS_SECTION,
    BLOCK_LENSING_MASS_GRIDS,
    BLOCK_METADATA,
    BLOCK_VELOCITY_DISPERSION_GRIDS,
)


def _json_ready(value: Any) -> Any:
    """Convert common scientific/Python values into JSON-serializable data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_string_dataset(group: h5py.Group, name: str, values: Any) -> None:
    """Write one variable-length UTF-8 string dataset."""

    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=string_dtype)


def _is_string_sequence(value: Any) -> bool:
    """Return whether a value should be serialized as a string dataset."""

    if isinstance(value, np.ndarray):
        return value.dtype.kind in {"O", "U", "S"}
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, str) for item in value)
    return False


def _write_mapping_value(group: h5py.Group, key: str, value: Any) -> None:
    """Write one mapping value as either a dataset or an attribute."""

    if _is_string_sequence(value):
        _write_string_dataset(group, key, value)
        return

    if isinstance(value, np.ndarray):
        group.create_dataset(key, data=value)
        return

    if isinstance(value, Mapping):
        group.attrs[f"{key}_json"] = json.dumps(_json_ready(value), sort_keys=True)
        return

    if isinstance(value, (list, tuple)):
        group.create_dataset(key, data=np.asarray(value))
        return

    if value is None:
        group.attrs[key] = ""
        return

    group.attrs[key] = value


def _write_mapping_group(parent: h5py.File | h5py.Group, name: str, values: Mapping[str, Any]) -> h5py.Group:
    """Create one group and serialize a flat mapping into it."""

    group = parent.create_group(name)
    for key, value in values.items():
        _write_mapping_value(group, key, value)
    return group


def _write_payload(handle: h5py.File, payload: CanonicalDatasetPayload) -> None:
    """Write all canonical top-level blocks into an open HDF5 handle."""

    metadata = _write_mapping_group(handle, BLOCK_METADATA, payload.metadata)
    metadata.attrs["provenance_json"] = json.dumps(_json_ready(payload.provenance), sort_keys=True)

    _write_mapping_group(handle, BLOCK_LENSES, payload.lenses)

    mass_group = _write_mapping_group(handle, BLOCK_LENSING_MASS_GRIDS, payload.lensing_mass_grids)
    if "s2_grid" in payload.velocity_dispersion_grids:
        _write_mapping_value(mass_group, "s2_grid", payload.velocity_dispersion_grids["s2_grid"])
    if "has_s2" in payload.velocity_dispersion_grids:
        _write_mapping_value(mass_group, "has_s2", payload.velocity_dispersion_grids["has_s2"])

    _write_mapping_group(handle, BLOCK_LENSING_CROSS_SECTION, payload.lensing_cross_section)

    velocity_group = handle.create_group(BLOCK_VELOCITY_DISPERSION_GRIDS)
    per_lens_group = _write_mapping_group(velocity_group, "per_lens_s2", payload.velocity_dispersion_grids)
    per_lens_group.attrs["source"] = f"/{BLOCK_LENSING_MASS_GRIDS}/s2_grid"


def write_canonical_dataset_payload(payload: CanonicalDatasetPayload, output_path: Path | str) -> Path:
    """Write one validated direct canonical payload using atomic replacement."""

    validate_canonical_dataset_payload(payload)

    resolved_output = Path(output_path).expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{resolved_output.name}.",
            suffix=".tmp",
            dir=resolved_output.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with h5py.File(temporary_path, "w") as handle:
            _write_payload(handle, payload)

        validate_canonical_hdf5(temporary_path)
        temporary_path.replace(resolved_output)
        return resolved_output
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


__all__ = [
    "write_canonical_dataset_payload",
]
