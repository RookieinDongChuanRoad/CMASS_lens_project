"""Migration-reference tests for the direct canonical pipeline.

These tests treat the old observation HDF5 file as a reference artifact only.
The direct pipeline must be able to match its catalog fields and sigma-count
distribution without making the old HDF5 file a runtime input.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from statistical_sl.data_preparation.direct_pipeline.catalogs import CmassSummaryCatalogReader
from statistical_sl.data_preparation.direct_pipeline.legacy_reference import (
    LegacyReferenceLens,
    num_sigma_distribution,
    read_legacy_reference_lenses,
)
from statistical_sl.data_preparation.direct_pipeline.policies import SigmaPolicy
from statistical_sl.data_preparation.direct_pipeline.records import BaseLensRecord, SigmaObservation
from statistical_sl.data_preparation.direct_pipeline.sigma_resolver import resolve_sigma_observations


def _candidate_project_roots() -> tuple[Path, ...]:
    """Return repository roots that may contain local, non-versioned raw data.

    The isolated worktree usually has source files but not the large ignored
    HDF5 artifacts.  When this checkout lives under ``.worktrees/``, the main
    checkout is a legitimate local reference-data candidate.  Tests still skip
    cleanly when neither root has the files.
    """

    worktree_root = Path(__file__).resolve().parents[2]
    roots = [worktree_root]
    if worktree_root.parent.name == ".worktrees":
        roots.append(worktree_root.parent.parent)
    return tuple(dict.fromkeys(roots))


def _reference_inputs_or_skip() -> tuple[Path, Path]:
    """Return matching summary-table and HDF5 paths, or skip if absent."""

    for root in _candidate_project_roots():
        summary_path = root / "workspace" / "data" / "raw" / "summary_table_deV.txt"
        reference_path = root / "workspace" / "data" / "raw" / "observations_deV_with_mass_grids.hdf5"
        if summary_path.exists() and reference_path.exists():
            return summary_path, reference_path
    pytest.skip("local legacy reference files are absent")


def _catalog_records_by_id(summary_path: Path) -> tuple[tuple[BaseLensRecord, ...], dict[str, BaseLensRecord]]:
    """Read the CMASS summary table through the new catalog-reader boundary."""

    catalog_result = CmassSummaryCatalogReader(summary_path, profile_name="devauc").read()
    return catalog_result.records, {record.lens_id: record for record in catalog_result.records}


def _synthetic_measurements_for_reference(reference_lenses: tuple[LegacyReferenceLens, ...]) -> tuple[SigmaObservation, ...]:
    """Create external sigma rows whose counts match the legacy HDF5 reference."""

    measurements: list[SigmaObservation] = []
    for row_index, reference_lens in enumerate(reference_lenses):
        if reference_lens.num_sigma == 0:
            continue
        if reference_lens.num_sigma == 1:
            measurements.append(
                SigmaObservation(
                    lens_id=reference_lens.lens_id,
                    sigma_kms=500.0 + row_index,
                    sigma_err_kms=10.0,
                )
            )
            continue
        if reference_lens.num_sigma == 2:
            measurements.extend(
                (
                    SigmaObservation(
                        lens_id=reference_lens.lens_id,
                        obs_tag="A",
                        sigma_kms=500.0 + row_index,
                        sigma_err_kms=10.0,
                    ),
                    SigmaObservation(
                        lens_id=reference_lens.lens_id,
                        obs_tag="B",
                        sigma_kms=600.0 + row_index,
                        sigma_err_kms=11.0,
                    ),
                )
            )
            continue
        raise AssertionError(f"Unsupported reference num_sigma={reference_lens.num_sigma}")
    return tuple(measurements)


def test_summary_catalog_fields_match_legacy_hdf5_subset() -> None:
    """The new catalog reader should recover the same lens facts as legacy HDF5."""

    summary_path, reference_path = _reference_inputs_or_skip()
    reference_lenses = read_legacy_reference_lenses(reference_path, limit=5)
    _, catalog_by_id = _catalog_records_by_id(summary_path)

    for reference_lens in reference_lenses:
        catalog_record = catalog_by_id[reference_lens.lens_id]
        assert catalog_record.z_lens == pytest.approx(reference_lens.z_lens)
        assert catalog_record.z_source == pytest.approx(reference_lens.z_source)
        assert catalog_record.theta_ein_arcsec == pytest.approx(reference_lens.theta_ein_arcsec)
        assert catalog_record.effective_radius_arcsec == pytest.approx(reference_lens.effective_radius_arcsec)
        assert catalog_record.log_stellar_mass == pytest.approx(reference_lens.log_stellar_mass)
        assert catalog_record.log_stellar_mass_err == pytest.approx(reference_lens.log_stellar_mass_err)


def test_summary_catalog_sigma_columns_are_not_promoted_to_observations() -> None:
    """Real CMASS summary-table sigma values should remain provenance only."""

    summary_path, _reference_path = _reference_inputs_or_skip()
    catalog_result = CmassSummaryCatalogReader(summary_path, profile_name="devauc").read()
    first_two_records = catalog_result.records[:2]
    first_lens_id = first_two_records[0].lens_id
    catalog_sigma = catalog_result.provenance.extra["untrusted_sigma_values"][first_lens_id]["sigma"]
    external_sigma = float(catalog_sigma) + 123.0

    resolution = resolve_sigma_observations(
        first_two_records,
        sigma_policy=SigmaPolicy(source_type="velocity_measurements_v1", missing_policy="num_sigma_zero"),
        accepted_measurements=(
            SigmaObservation(
                lens_id=first_lens_id,
                sigma_kms=external_sigma,
                sigma_err_kms=10.0,
            ),
        ),
    )

    assert resolution.records[0].sigma_observations[0].sigma_kms == pytest.approx(external_sigma)
    assert resolution.records[0].sigma_observations[0].sigma_kms != pytest.approx(catalog_sigma)
    assert resolution.records[1].num_sigma == 0


def test_external_measurements_can_reproduce_legacy_num_sigma_distribution() -> None:
    """The resolver can match legacy counts without taking the legacy HDF5 as input."""

    summary_path, reference_path = _reference_inputs_or_skip()
    reference_lenses = read_legacy_reference_lenses(reference_path)
    catalog_records, catalog_by_id = _catalog_records_by_id(summary_path)
    base_lenses = tuple(catalog_by_id[reference_lens.lens_id] for reference_lens in reference_lenses)
    synthetic_measurements = _synthetic_measurements_for_reference(reference_lenses)

    resolution = resolve_sigma_observations(
        base_lenses,
        sigma_policy=SigmaPolicy(source_type="velocity_measurements_v1", missing_policy="num_sigma_zero"),
        accepted_measurements=synthetic_measurements,
    )

    assert len(catalog_records) >= len(reference_lenses)
    assert Counter(record.num_sigma for record in resolution.records) == Counter(
        reference_lens.num_sigma for reference_lens in reference_lenses
    )
    assert dict(num_sigma_distribution(reference_lenses)) == {0: 10, 1: 10, 2: 3}
