"""
Unit tests for reusable canonical-context preprocessing helpers.

The helpers tested here sit below concrete scientific models.  They should know
about canonical dataset shapes and metadata, but they should not import or
construct CMASS model contexts.  That boundary is what lets future models reuse
canonical dataset handling without inheriting CMASS formulas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cmass_lens_inference.canonical_dataset import (
    CanonicalCrossSectionGrid,
    CanonicalInferenceDataset,
    CanonicalLenses,
    CanonicalLensingMassGrids,
    CanonicalMetadata,
    CanonicalSigmaGrid,
    CanonicalVelocityDispersionGrids,
)
from cmass_lens_inference.canonical_context import (
    canonical_dataset_metadata,
    interpolate_lensing_mass_grids,
    normalize_sigma_grid,
    shared_gamma_axis,
)


def _minimal_dataset() -> CanonicalInferenceDataset:
    """Build one tiny canonical dataset object without touching HDF5."""

    gamma_grid = np.asarray(
        [
            [1.2, 2.0, 2.8],
            [1.3, 2.1, 2.7],
        ],
        dtype=np.float64,
    )
    return CanonicalInferenceDataset(
        path=Path("/tmp/canonical.hdf5"),
        metadata=CanonicalMetadata(
            schema_version="canonical_inference_dataset_v1",
            unit_convention="h_units_v1",
            h_ref=0.7,
            profile_name="devauc",
            mass_definition_label="m5_hinvkpc",
            mass_radius_kpc=5.0,
            cosmology_h0=70.0,
            cosmology_omega_m=0.3,
            capabilities=frozenset({"z.capability", "a.capability"}),
        ),
        lenses=CanonicalLenses(
            lens_id=("lens-a", "lens-b"),
            z_d=np.asarray([0.5, 0.6]),
            z_s=np.asarray([1.6, 1.8]),
            log_mstar_obs=np.asarray([11.0, 11.1]),
            log_mstar_err=np.asarray([0.05, 0.06]),
            log_re_obs=np.asarray([0.6, 0.7]),
            n_obs=np.asarray([4.0, 4.0]),
            theta_e_obs=np.asarray([1.0, 1.1]),
            num_sigma=np.asarray([0, 0], dtype=np.int64),
            sigma_obs=np.zeros((2, 2), dtype=np.float64),
            sigma_err=np.ones((2, 2), dtype=np.float64),
        ),
        mass_grids=CanonicalLensingMassGrids(
            gamma_grid=gamma_grid,
            log_enclosed_mass_grid=np.asarray(
                [
                    [11.2, 11.0, 10.8],
                    [11.4, 11.1, 10.9],
                ],
                dtype=np.float64,
            ),
            dmass_dthetaein_grid=np.asarray(
                [
                    [-2.0, -1.5, -1.0],
                    [-2.2, -1.6, -1.2],
                ],
                dtype=np.float64,
            ),
            s2_grid=np.asarray(
                [
                    [0.7, 0.8, 0.9],
                    [0.6, 0.7, 0.8],
                ],
                dtype=np.float64,
            ),
            has_s2=np.asarray([1, 1], dtype=np.int64),
        ),
        cross_section=CanonicalCrossSectionGrid(
            theta_e_axis=np.asarray([0.0, 1.0]),
            gamma_axis=np.asarray([1.2, 2.0, 2.8]),
            cross_section_grid=np.ones((2, 3), dtype=np.float64),
            boundary_policy="zero_outside_theta_clip_gamma",
        ),
        velocity_dispersion=CanonicalVelocityDispersionGrids(
            per_lens_s2_grid=None,
            per_lens_has_s2=None,
            fp_within_re=None,
            population_sigma_unit=None,
        ),
    )


def test_canonical_dataset_metadata_is_sorted_and_json_ready() -> None:
    """Metadata extraction should be model-neutral and deterministic."""

    metadata = canonical_dataset_metadata(_minimal_dataset())

    assert metadata == {
        "canonical_dataset_path": "/tmp/canonical.hdf5",
        "canonical_schema_version": "canonical_inference_dataset_v1",
        "canonical_capabilities": ("a.capability", "z.capability"),
        "canonical_profile_name": "devauc",
        "canonical_mass_definition_label": "m5_hinvkpc",
    }


def test_interpolate_lensing_mass_grids_handles_per_lens_gamma_axes() -> None:
    """Shared interpolation helper should not assume CMASS context fields."""

    dataset = _minimal_dataset()
    target_gamma_axis = shared_gamma_axis(dataset.mass_grids.gamma_grid, n_points=5)
    mass_grid, derivative_grid, s2_grid = interpolate_lensing_mass_grids(
        dataset.mass_grids,
        target_gamma_axis,
    )

    assert target_gamma_axis.shape == (5,)
    assert mass_grid.shape == (2, 5)
    assert derivative_grid.shape == (2, 5)
    assert s2_grid.shape == (2, 5)
    assert mass_grid[0, 0] == np.float64(11.2)
    assert mass_grid[1, -1] == np.float64(10.9)


def test_normalize_sigma_grid_injects_missing_axes() -> None:
    """Canonical sigma grids should normalize to gamma, zd, logRe, n order."""

    sigma_grid = CanonicalSigmaGrid(
        gamma_axis=np.asarray([1.2, 2.0]),
        zd_axis=np.asarray([0.0]),
        log_re_axis=np.asarray([0.5, 0.7, 0.9]),
        sigma_unit_grid=np.ones((2, 3), dtype=np.float64),
        n_axis=np.asarray([4.0]),
    )

    gamma_axis, zd_axis, log_re_axis, n_axis, values, has_n_axis = normalize_sigma_grid(
        sigma_grid,
        profile_fixed_n=4.0,
    )

    assert gamma_axis.shape == (2,)
    assert zd_axis.shape == (1,)
    assert log_re_axis.shape == (3,)
    assert n_axis.tolist() == [4.0]
    assert values.shape == (2, 1, 3, 1)
    assert has_n_axis == 0


def test_normalize_sigma_grid_preserves_real_redshift_axis_for_3d_table() -> None:
    """A 3D population table with a real redshift axis must not be read as n-dependent."""

    raw_values = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    sigma_grid = CanonicalSigmaGrid(
        gamma_axis=np.asarray([1.2, 2.0]),
        zd_axis=np.asarray([0.05, 0.225, 0.4]),
        log_re_axis=np.asarray([0.45, 0.825, 1.2]),
        sigma_unit_grid=raw_values,
        n_axis=np.asarray([4.0]),
    )

    _gamma_axis, _zd_axis, _log_re_axis, n_axis, values, has_n_axis = normalize_sigma_grid(
        sigma_grid,
        profile_fixed_n=4.0,
    )

    assert n_axis.tolist() == [4.0]
    assert values.shape == (2, 3, 3, 1)
    np.testing.assert_array_equal(values[..., 0], raw_values)
    assert has_n_axis == 0
