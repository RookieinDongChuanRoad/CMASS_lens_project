from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from statistical_sl.inference.config import load_runtime_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DEVAUC_CONFIG = REPOSITORY_ROOT / "workspace/configs/inference/cmass/devauc.yaml"
CURRENT_DEVAUC_CANONICAL_DATASET = (
    REPOSITORY_ROOT / "workspace/data/canonical/inference_dataset_devauc_slit_m5_hunits_v1.hdf5"
)
DEVAUC_SIGMA_BUNDLE_20260429 = (
    REPOSITORY_ROOT / "workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5"
)


def test_current_cmass_devauc_config_enables_20260429_fp_prior_contract() -> None:
    """The maintained CMASS de Vaucouleurs config should keep the 2024-04-29 FP prior active.

    The historical 2024-04-29 run is the relevant FP-prior science contract for
    the current CMASS de Vaucouleurs configuration.  The modern config no longer
    names the old raw observation and sigma-bundle files directly, but enabling
    the FP prior remains a scientific choice rather than a path-layout detail.
    This test prevents the config from silently drifting back to a no-FP run.
    """

    runtime_config = load_runtime_config(CURRENT_DEVAUC_CONFIG)

    assert runtime_config.fp_prior.enabled is True
    assert np.isclose(runtime_config.fp_prior.mu_v_prior, 2.34548)
    assert np.isclose(runtime_config.fp_prior.mu_v_error, 0.00611)
    assert np.isclose(runtime_config.fp_prior.beta_v_prior, 0.176)
    assert np.isclose(runtime_config.fp_prior.beta_v_error, 0.011)
    assert np.isclose(runtime_config.fp_prior.fiducial_scatter, 0.075)
    assert np.isclose(runtime_config.fp_prior.scatter_error, 0.003)


def test_current_canonical_dataset_embeds_the_20260429_fp_within_re_sigma_grid() -> None:
    """The canonical dataset should preserve the old within-Re FP sigma grid exactly.

    The 2024-04-29 run read the FP prior's velocity-dispersion grid from
    ``jeans_deV_sigma_bundle.h5`` at ``within_re/m5_hinvkpc``.  The current
    canonical dataset stores the same contract under
    ``velocity_dispersion_grids/fp_within_re`` so inference can use one canonical
    HDF5 input instead of mixing raw observations, cross sections, and sigma
    bundle files at runtime.
    """

    with (
        h5py.File(CURRENT_DEVAUC_CANONICAL_DATASET, "r") as canonical_dataset,
        h5py.File(DEVAUC_SIGMA_BUNDLE_20260429, "r") as sigma_bundle,
    ):
        canonical_group = canonical_dataset["velocity_dispersion_grids/fp_within_re"]
        bundle_group = sigma_bundle["within_re/m5_hinvkpc"]

        for dataset_name in ("gamma_axis", "log_re_kpc_axis", "s_unit_grid"):
            np.testing.assert_array_equal(
                canonical_group[dataset_name][...],
                bundle_group[dataset_name][...],
                err_msg=f"{dataset_name} drifted from the 2024-04-29 FP sigma-bundle contract.",
            )
