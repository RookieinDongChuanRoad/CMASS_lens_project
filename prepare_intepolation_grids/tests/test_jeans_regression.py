"""Regression and policy tests for the velocity-dispersion grid calculation.

Two distinct behaviors matter now:

1. We must be able to reproduce the historical `make_jeans_grid.py` behavior
   for the free-Sersic branch. That verifies the new code still understands the
   old reference implementation correctly.
2. The production code has intentionally diverged from that historical script
   by using a fixed 1.6 arcsec aperture width while keeping seeing fixed at
   0.9 arcsec. That policy change must also be tested explicitly.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from astropy.constants import G, M_sun, kpc
from astropy.cosmology import FlatLambdaCDM
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import powerlaw

from interpolation_grids.config import (
    APERTURE_HEIGHT_ARCSEC,
    BOSS_CIRCULAR_APERTURE_POLICY,
    DEFAULT_INPUT_FILENAMES,
    DEFAULT_APERTURE_WIDTH_ARCSEC,
    DEFAULT_PRODUCTION_APERTURE_POLICY,
    DEFAULT_RADIAL_GRID_SIZE,
    GAMMA_GRID,
    SEEING_FWHM_ARCSEC,
)
from interpolation_grids.io.hdf5 import build_galaxy_inputs
from interpolation_grids.models import AperturePolicy
from interpolation_grids.physics.jeans import compute_s2_grid, compute_sigma_unit_grid, kpc_per_arcsec, uses_devaucouleurs_branch


DATA_ROOT = "/Users/liurongfu/Work/CMASS_lens_project/data/raw"
COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)
SIGMA2_TO_KM2_PER_S2 = (G * M_sun / kpc).to("km2 / s2").value


def _resolve_existing_raw_file(*candidates: str) -> str:
    """Return the first raw-data path that exists on this machine.

    The historical regression tests originally used legacy `with_m5_grids`
    files. The current local environment only ships the canonical
    `with_mass_grids` files, so the tests need a deterministic fallback rather
    than assuming one specific storage layout.
    """

    for candidate in candidates:
        path = Path(DATA_ROOT) / candidate
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"None of the expected raw files exist: {candidates}")


def test_default_input_filenames_use_canonical_mass_grid_names() -> None:
    """The public prepare defaults should no longer advertise `with_m5_grids`.

    This test locks the canonical raw-observation naming migration. The old
    filenames remain compatible as legacy inputs, but anything described as a
    default path should now use the definition-agnostic `with_mass_grids`
    surface so future m10 runs do not keep baking `m5` into user-visible file
    names.
    """

    assert DEFAULT_INPUT_FILENAMES == (
        "observations_deV_with_mass_grids.hdf5",
        "observations_with_mass_grids_all.hdf5",
    )


def test_uses_devaucouleurs_branch_accepts_new_and_legacy_canonical_filenames() -> None:
    """Profile detection must not be tied to one historical filename string."""

    assert uses_devaucouleurs_branch("observations_deV_with_m5_grids.hdf5") is True
    assert uses_devaucouleurs_branch("observations_deV_with_mass_grids.hdf5") is True
    assert uses_devaucouleurs_branch("observations_with_mass_grids_all.hdf5") is False
    assert uses_devaucouleurs_branch("observations_with_m5_grids_all.hdf5") is False


def _legacy_reference_s2_grid(galaxy, gamma_grid: np.ndarray) -> np.ndarray:
    """Reproduce the historical `make_jeans_grid.py` calculation path.

    This helper intentionally uses the aperture width stored in the input file,
    because the legacy script did the same. It is a test-only compatibility
    oracle, not the current production policy.
    """

    physical_kpc_per_arcsec = COSMOLOGY.kpc_proper_per_arcmin(galaxy.zd).value / 60.0
    aperture_kpc = [
        APERTURE_HEIGHT_ARCSEC * physical_kpc_per_arcsec,
        galaxy.aperture_width_arcsec * physical_kpc_per_arcsec,
    ]
    seeing_kpc = SEEING_FWHM_ARCSEC * physical_kpc_per_arcsec
    radial_anchor_kpc = galaxy.re_arcsec * physical_kpc_per_arcsec
    radial_grid = np.logspace(
        np.log10(radial_anchor_kpc) - 3.0,
        np.log10(radial_anchor_kpc) + 3.0,
        DEFAULT_RADIAL_GRID_SIZE,
    )

    output = np.zeros_like(gamma_grid, dtype=float)
    for index, gamma in enumerate(gamma_grid):
        normalization = 1.0 / powerlaw.M2d(5.0, gamma)
        enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
        output[index] = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            [radial_anchor_kpc, galaxy.nser],
            tracer_profiles.sersic,
            seeing=seeing_kpc,
        ) * SIGMA2_TO_KM2_PER_S2
    return output


def _policy_s2_grid(galaxy, gamma_grid: np.ndarray, aperture_width_arcsec: float) -> np.ndarray:
    """Compute `s2_grid` for an explicit aperture policy.

    The helper mirrors the production implementation but keeps the aperture
    width overridable so tests can compare the old 0.8 arcsec behavior against
    the new 1.6 arcsec business rule while leaving seeing fixed.
    """

    physical_kpc_per_arcsec = kpc_per_arcsec(galaxy.zd)
    aperture_kpc = [
        APERTURE_HEIGHT_ARCSEC * physical_kpc_per_arcsec,
        aperture_width_arcsec * physical_kpc_per_arcsec,
    ]
    seeing_kpc = SEEING_FWHM_ARCSEC * physical_kpc_per_arcsec

    if uses_devaucouleurs_branch(galaxy.source_filename):
        radial_anchor_kpc = galaxy.reff_dev_arcsec * physical_kpc_per_arcsec
        tracer_parameters = radial_anchor_kpc
        tracer_profile = tracer_profiles.deVaucouleurs
    else:
        radial_anchor_kpc = galaxy.re_arcsec * physical_kpc_per_arcsec
        tracer_parameters = (radial_anchor_kpc, galaxy.nser)
        tracer_profile = tracer_profiles.sersic

    radial_grid = np.logspace(
        np.log10(radial_anchor_kpc) - 3.0,
        np.log10(radial_anchor_kpc) + 3.0,
        DEFAULT_RADIAL_GRID_SIZE,
    )

    output = np.zeros_like(gamma_grid, dtype=float)
    for index, gamma in enumerate(gamma_grid):
        normalization = 1.0 / powerlaw.M2d(5.0, gamma)
        enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
        output[index] = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            tracer_parameters,
            tracer_profile,
            seeing=seeing_kpc,
        ) * SIGMA2_TO_KM2_PER_S2
    return output


def _explicit_policy_s2_grid(galaxy, gamma_grid: np.ndarray, aperture_policy: AperturePolicy) -> np.ndarray:
    """Compute an explicit policy variant used to validate the public helper.

    This helper mirrors the production code path but leaves the aperture
    representation fully explicit so the tests can validate both rectangular
    and circular BOSS policies against the shared public API.
    """

    physical_kpc_per_arcsec = kpc_per_arcsec(galaxy.zd)
    if aperture_policy.shape == "circular":
        aperture_kpc = float(aperture_policy.radius_arcsec * physical_kpc_per_arcsec)
    else:
        aperture_kpc = [
            float(aperture_policy.height_arcsec * physical_kpc_per_arcsec),
            float(aperture_policy.width_arcsec * physical_kpc_per_arcsec),
        ]
    seeing_kpc = float(aperture_policy.seeing_fwhm_arcsec * physical_kpc_per_arcsec)

    if uses_devaucouleurs_branch(galaxy.source_filename):
        radial_anchor_kpc = galaxy.reff_dev_arcsec * physical_kpc_per_arcsec
        tracer_parameters = radial_anchor_kpc
        tracer_profile = tracer_profiles.deVaucouleurs
    else:
        radial_anchor_kpc = galaxy.re_arcsec * physical_kpc_per_arcsec
        tracer_parameters = (radial_anchor_kpc, galaxy.nser)
        tracer_profile = tracer_profiles.sersic

    radial_grid = np.logspace(
        np.log10(radial_anchor_kpc) - 3.0,
        np.log10(radial_anchor_kpc) + 3.0,
        DEFAULT_RADIAL_GRID_SIZE,
    )

    output = np.zeros_like(gamma_grid, dtype=float)
    for index, gamma in enumerate(gamma_grid):
        normalization = 1.0 / powerlaw.M2d(5.0, gamma)
        enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
        output[index] = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            tracer_parameters,
            tracer_profile,
            seeing=seeing_kpc,
        ) * SIGMA2_TO_KM2_PER_S2
    return output


def test_sersic_s2_grid_matches_legacy_reference_script_path() -> None:
    """The legacy helper should still reproduce the old 0.8 arcsec recipe."""

    input_path = _resolve_existing_raw_file(
        "observations_with_m5_grids_all.hdf5",
        "observations_with_mass_grids_all.hdf5",
    )
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )
    actual = _legacy_reference_s2_grid(galaxy=galaxy, gamma_grid=GAMMA_GRID)
    expected = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=galaxy.aperture_width_arcsec,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_production_sersic_s2_grid_uses_fixed_1p6_arcsec_aperture() -> None:
    """Production Sersic output should follow the new 1.6 arcsec policy."""

    input_path = _resolve_existing_raw_file(
        "observations_with_m5_grids_all.hdf5",
        "observations_with_mass_grids_all.hdf5",
    )
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )
        stored_production = handle[group_name]["s2_grid"][:]

    actual = compute_s2_grid(galaxy=galaxy, gamma_grid=GAMMA_GRID)
    expected_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=DEFAULT_APERTURE_WIDTH_ARCSEC,
    )
    old_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=galaxy.aperture_width_arcsec,
    )

    np.testing.assert_allclose(actual, expected_policy, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(actual, stored_production, rtol=1e-8, atol=1e-12)
    assert np.max(np.abs(actual - old_policy) / old_policy) > 0.05


def test_production_devaucouleurs_s2_grid_uses_fixed_1p6_arcsec_aperture() -> None:
    """The deV branch should also follow the new fixed-aperture policy."""

    input_path = _resolve_existing_raw_file(
        "observations_deV_with_m5_grids.hdf5",
        "observations_deV_with_mass_grids.hdf5",
    )
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )
        stored_production = handle[group_name]["s2_grid"][:]

    actual = compute_s2_grid(galaxy=galaxy, gamma_grid=GAMMA_GRID)
    expected_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=DEFAULT_APERTURE_WIDTH_ARCSEC,
    )
    old_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=galaxy.aperture_width_arcsec,
    )

    np.testing.assert_allclose(actual, expected_policy, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(actual, stored_production, rtol=1e-8, atol=1e-12)
    assert np.max(np.abs(actual - old_policy) / old_policy) > 0.05


def test_compute_sigma_unit_grid_passes_float_aperture_for_boss_circular_policy(monkeypatch) -> None:
    """The BOSS circular policy should call `spherical_jeans` with a float aperture.

    This test protects the public contract of the new aperture-policy
    abstraction: circular apertures must be represented as a scalar radius so
    the downstream dependency evaluates the correct geometry.
    """

    captured = {}

    def _fake_sigma2(*args, **kwargs):
        captured["aperture"] = args[1]
        captured["seeing"] = kwargs["seeing"]
        return 1.0

    monkeypatch.setattr("interpolation_grids.physics.jeans.sigma_model.sigma2", _fake_sigma2)

    values = compute_sigma_unit_grid(
        profile_name="devauc",
        gamma_grid=np.asarray([2.0], dtype=float),
        zd=0.6,
        re_kpc=5.0,
        aperture_policy=BOSS_CIRCULAR_APERTURE_POLICY,
    )

    assert values.shape == (1,)
    assert isinstance(captured["aperture"], float)
    assert captured["aperture"] > 0.0
    assert captured["seeing"] > 0.0


def test_boss_sersic_s2_grid_uses_circular_one_arcsec_aperture() -> None:
    """BOSS Sersic grids should use the circular 1 arcsec aperture policy."""

    input_path = _resolve_existing_raw_file(
        "observations_with_m5_grids_all.hdf5",
        "observations_with_mass_grids_all.hdf5",
    )
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )

    actual = compute_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_policy=BOSS_CIRCULAR_APERTURE_POLICY,
    )
    expected = _explicit_policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_policy=BOSS_CIRCULAR_APERTURE_POLICY,
    )
    old_rectangular_policy = _explicit_policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_policy=DEFAULT_PRODUCTION_APERTURE_POLICY,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    assert np.max(np.abs(actual - old_rectangular_policy) / old_rectangular_policy) > 0.01


def test_boss_devaucouleurs_s2_grid_uses_circular_one_arcsec_aperture() -> None:
    """BOSS deV grids should also use the circular 1 arcsec aperture policy."""

    input_path = _resolve_existing_raw_file(
        "observations_deV_with_m5_grids.hdf5",
        "observations_deV_with_mass_grids.hdf5",
    )
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )

    actual = compute_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_policy=BOSS_CIRCULAR_APERTURE_POLICY,
    )
    expected = _explicit_policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_policy=BOSS_CIRCULAR_APERTURE_POLICY,
    )
    old_rectangular_policy = _explicit_policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_policy=DEFAULT_PRODUCTION_APERTURE_POLICY,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    assert np.max(np.abs(actual - old_rectangular_policy) / old_rectangular_policy) > 0.01
