# CMASS Lens-Only Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production `cmass_lens_only` inference model that mirrors the Sonnenfeld 2024 lens-only comparison: fit the already-observed CMASS lens sample directly, without lensing selection correction, source-redshift population inference, lens-finding probability, cross-section weighting, or FP prior.

**Architecture:** Implement `cmass_lens_only` as a new concrete registry model under `Bayesian_inference/src/cmass_lens_inference/models/`, not as a switch inside `cmass`. Reuse the existing CMASS canonical preprocessing and per-lens mass / velocity-dispersion grids, but wrap the base context with lens-only stellar-mass observation density and route evaluation through a dedicated Numba posterior. The model keeps the current CMASS mass and gamma population laws but changes the target distribution from parent-population-with-selection to observed-lens-sample.

**Tech Stack:** Python dataclasses, YAML config, HDF5 canonical datasets, Numba kernels, emcee sampler, pytest, `cmass_lens` conda environment.

---

## Scientific Target

The model should encode this likelihood for each observed lens:

```text
L_i(theta) =
  integral dlogMstar dgamma
    P(logMstar_obs_i | logMstar)
  * P_lens(logMstar | mu_mstar_lens, sigma_mstar_lens)
  * P(m5_obs_i(gamma) | logMstar, observed_size_i, theta)
  * P(gamma | logMstar, observed_size_i, theta)
  * P(sigma_obs_i | sigma_model_i(m5_obs_i(gamma), gamma))
  * |dm5 / dtheta_E|_i(gamma)
```

The total posterior under uniform box priors is:

```text
logp(theta) = sum_i log L_i(theta)
```

This model intentionally removes:

- `P_find`
- lensing cross-section weight
- selection normalization `z_norm`
- `mu_zs`, `sigma_zs`
- `theta0`, `loga`
- FP prior and FP population-summary reducer
- parent redshift / source-redshift likelihood terms

The first implementation should reject `fp_prior.enabled: true` for `cmass_lens_only`. A later explicit model can add `cmass_lens_only_fp_prior` if that becomes a real scientific target.

## Component Assembly and Kernel Reuse Contract

`cmass_lens_only` should remain a concrete model assembled from the existing component declarations where they carry real scientific structure.  It should not expose a `model.components` config surface, and it should not introduce a separate observed-velocity-dispersion likelihood component in this first implementation.  The velocity-dispersion likelihood stays explicitly assembled inside the dedicated posterior kernel, matching the current CMASS posterior style and keeping all likelihood-ordering details visible in one hot path.

The model assembly should include:

- `observations.lens_sample`
  - Reuse `lens_sample_component`.
  - Requires `lens_observations.v1`.
  - Supplies observed lens covariates such as `log_mstar_obs`, `log_mstar_err`, `log_re_obs`, `n_obs`, `sigma_obs`, `sigma_err`, and `num_sigma`.
- `population.stellar_mass_function.gaussian_lens_sample`
  - Add this small component declaration because lens-only needs a Gaussian stellar-mass distribution for the already-observed lens sample, not the parent-population skew-normal used by selection-corrected CMASS.
  - Owns `mu_mstar_lens` and `sigma_mstar_lens`.
  - Requires the shared `normal_pdf` kernel.
- `population.size_relation.linear`
  - Reuse `linear_size_relation_component`.
  - Supplies the deterministic `delta_r_grid` term used by the aperture-mass relation.
- `population.aperture_mass_relation.gaussian_linear`
  - Reuse `gaussian_linear_aperture_mass_component`.
  - Owns `mu5_0`, `beta5`, `xi5`, and `sigma5`.
  - Requires the shared `gaussian_linear_mass_mean` kernel.
- `population.gamma_relation.sigma_star_linear`
  - Reuse `sigma_star_linear_gamma_component`.
  - Owns `mu_gamma_0`, `beta_sigma_star_gamma`, and `sigma_gamma`.
  - Requires the shared `sigma_star_linear_gamma_mean` kernel.

The model assembly must not include:

- `population.stellar_mass_function.skewnormal`
- `population.source_redshift.truncated_nonnegative_gaussian`
- `lensing.cross_section.theta_gamma`
- `selection.discovery_probability`
- any FP-prior reducer or FP-prior component
- any new `observed_velocity_dispersion_likelihood` component

The dedicated posterior kernel should reuse these shared Numba kernels directly:

- `normal_pdf`
- `trapezoid_1d`
- `gaussian_linear_mass_mean`
- `sigma_star_linear_gamma_mean`
- `sigma_model_from_s2`
- `observed_sigma_likelihood`

The posterior kernel must not call or import the default `cmass.posterior` helper functions.  In particular, it should not import `cmass_gamma_population_mean`; use `sigma_star_linear_gamma_mean` from the shared population-kernel module instead.  This keeps `cmass_lens_only` dependent on reusable kernels, not on the implementation details of the default CMASS posterior.

The strict model-level canonical capability contract is:

- `lens_observations.v1`
- `lensing_mass_grids.v1`
- `velocity_dispersion.per_lens_s2.v1`

The current canonical reader still requires the HDF5 `lensing_cross_section` block to exist as a schema-level fact, but `cmass_lens_only` must not declare `lensing_cross_section.theta_gamma_grid.v1` as a required capability and must not use those values in the likelihood.  The cross-section invariance test below is the guardrail for this distinction.

## File Structure

Create:

- `Bayesian_inference/src/cmass_lens_inference/components/population/stellar_mass_function/gaussian_lens_sample.py`
  - Reusable component declaration for the observed lens-sample Gaussian stellar-mass distribution.
- `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/__init__.py`
  - Public package exports for the new model.
- `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/assembly.py`
  - Human-authored `ModelSpec`, parameter list, metadata, and required capabilities.
- `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/context.py`
  - Small wrapper context around the existing `CMASSModelContext`.
- `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/preprocessing.py`
  - Canonical dataset loader with lens-only capability requirements and stellar-mass observation density construction.
- `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/runtime.py`
  - Runtime adapter for the registry.
- `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/posterior.py`
  - Dedicated lens-only Numba likelihood.
- `Bayesian_inference/configs/cmass_lens_only.yaml`
  - Real-run config for the current h-units CMASS dataset.
- `Bayesian_inference/tests/test_cmass_lens_only_model.py`
  - Focused tests for model contract and numerical behavior.
- `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/README.md`
  - Near-code scientific contract and limitations.

Modify:

- `Bayesian_inference/src/cmass_lens_inference/model_registry.py`
  - Register `cmass_lens_only`.
- `Bayesian_inference/src/cmass_lens_inference/components/population/stellar_mass_function/__init__.py`
  - Export the observed lens-sample Gaussian stellar-mass component.
- `Bayesian_inference/tests/test_model_registry_config.py`
  - Extend registry coverage and expected model-name list.
- `Bayesian_inference/tests/conftest.py`
  - Add fixture helpers for lens-only box priors and synthetic configs.
- `Bayesian_inference/docs/model_refactor_progress.md`
  - Record that lens-only is a separate concrete model, not a `cmass` component switch.

Do not modify:

- `prepare_dataset/`
- `Posterior_predictive_test/`
- existing `cmass` scientific semantics
- existing real canonical datasets

---

### Task 1: Add Lens-Only Registry Contract Tests

**Files:**
- Create: `Bayesian_inference/tests/test_cmass_lens_only_model.py`
- Modify: `Bayesian_inference/tests/conftest.py`

- [ ] **Step 1: Add fixture helpers for lens-only config payloads**

Add this helper near `_default_box_prior_config()` in `Bayesian_inference/tests/conftest.py`:

```python
def _lens_only_box_prior_config() -> dict[str, list[float]]:
    """
    Return the public-name prior mapping for the CMASS lens-only model.

    The lens-only model keeps the current CMASS h-unit mass and
    sigma-star-dependent gamma laws, but replaces source/selection parameters
    with a Gaussian distribution for the observed lens stellar-mass sample.
    """

    return {
        "mu_mstar_lens": [10.0, 12.5],
        "sigma_mstar_lens": [1.0e-3, 1.0],
        "mu5h_0": [9.0, 12.0],
        "beta5h": [-3.0, 3.0],
        "xi5h": [-3.0, 3.0],
        "sigma5h": [1.0e-2, 0.2],
        "mu_gamma_0": [1.5, 2.5],
        "beta_sigma_star_gamma": [-3.0, 3.0],
        "sigma_gamma": [1.0e-3, 0.5],
    }
```

Add this helper below `_cmass_model_config()`:

```python
def _cmass_lens_only_model_config() -> dict:
    """
    Return the registry-backed lens-only model section for fixtures.

    Keeping this helper separate from `_cmass_model_config` prevents tests from
    accidentally treating lens-only behavior as a component switch on `cmass`.
    """

    return {"name": "cmass_lens_only"}
```

- [ ] **Step 2: Add a synthetic lens-only config fixture**

Add this fixture near `synthetic_config_path` in `Bayesian_inference/tests/conftest.py`:

```python
@pytest.fixture
def synthetic_lens_only_config_path(
    tmp_path: Path,
    synthetic_hunit_observation_file: Path,
    synthetic_cross_section_file: Path,
) -> Path:
    """
    Create a compact config for the `cmass_lens_only` model.

    The fixture still writes a canonical dataset with a cross-section block
    because the current canonical reader requires that block. The lens-only
    posterior must prove through tests that it ignores those values.
    """

    path = tmp_path / "synthetic_cmass_lens_only.yaml"
    canonical_dataset_path = _write_canonical_dataset_from_legacy_inputs(
        output_path=tmp_path / "synthetic_cmass_lens_only_canonical.hdf5",
        observation_path=synthetic_hunit_observation_file,
        cross_section_path=synthetic_cross_section_file,
        profile_name="sersic",
    )
    config = {
        "profile": {"name": "sersic"},
        "unit_convention": "h_units_v1",
        "model": _cmass_lens_only_model_config(),
        "data": {
            "inference_dataset_path": str(canonical_dataset_path),
        },
        "box_prior": _lens_only_box_prior_config(),
        "sampling": {
            "random_seed": 7,
            "n_walkers": 24,
            "n_steps": 3,
            "burn_in": 1,
            "initial_center": {
                "mu_mstar_lens": 11.0,
                "sigma_mstar_lens": 0.15,
                "mu5h_0": 11.17,
                "beta5h": 0.59,
                "xi5h": -0.11,
                "sigma5h": 0.06,
                "mu_gamma_0": 1.99,
                "beta_sigma_star_gamma": 0.24,
                "sigma_gamma": 0.149,
            },
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 200,
            "mstar_points": 200,
            "normalization_samples": 128,
        },
        "cosmology": {
            "h0": 70.0,
            "omega_m": 0.3,
        },
        "runtime": {
            "checkpoint_every": 1,
            "parallel_strategy": "auto",
            "progress": False,
            "progress_summary_every": 1,
            "show_stage_timing": True,
            "disable_hdf5_file_locking": False,
            "num_threads": 0,
            "reserve_cores": 2,
        },
        "output": {
            "root_dir": str(tmp_path / "outputs"),
            "run_label": "synthetic-cmass-lens-only",
            "overwrite_latest": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
```

- [ ] **Step 3: Write registry and schema tests before implementation**

Create `Bayesian_inference/tests/test_cmass_lens_only_model.py` with:

```python
"""Tests for the CMASS lens-only model."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.numba_backend.likelihood_engine import (
    build_compiled_model as build_numba_model,
    log_prob as numba_log_prob,
)


def test_cmass_lens_only_is_registered_as_concrete_model() -> None:
    """The registry should expose lens-only as its own scientific model."""

    model_definition = get_model_definition("cmass_lens_only")

    assert model_definition.name == "cmass_lens_only"
    assert model_definition.backend_kernel == "cmass_lens_only"
    assert model_definition.required_capabilities == (
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    )
    assert CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1 not in (
        model_definition.required_capabilities
    )
    assert model_definition.optional_capabilities == ()


def test_cmass_lens_only_config_has_lens_only_parameter_schema(
    synthetic_lens_only_config_path: Path,
) -> None:
    """Lens-only should drop source-redshift and discovery parameters."""

    runtime_config = load_runtime_config(synthetic_lens_only_config_path)

    assert runtime_config.model.name == "cmass_lens_only"
    assert runtime_config.parameter_schema.public_parameter_names == (
        "mu_mstar_lens",
        "sigma_mstar_lens",
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
    )
    assert "mu_zs" not in runtime_config.parameter_schema.public_parameter_names
    assert "sigma_zs" not in runtime_config.parameter_schema.public_parameter_names
    assert "theta0" not in runtime_config.parameter_schema.public_parameter_names
    assert "loga" not in runtime_config.parameter_schema.public_parameter_names
    assert runtime_config.parameter_schema.model_metadata["selection_correction"] is False


def test_cmass_lens_only_rejects_fp_prior(
    synthetic_lens_only_config_path: Path,
) -> None:
    """The first lens-only implementation should not silently mix FP prior semantics."""

    payload = yaml.safe_load(synthetic_lens_only_config_path.read_text(encoding="utf-8"))
    payload["fp_prior"] = {"enabled": True}
    config_path = synthetic_lens_only_config_path.parent / "lens_only_fp_prior.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(config_path)

    with pytest.raises(ValueError, match="cmass_lens_only.*fp_prior"):
        build_numba_model(runtime_config)
```

- [ ] **Step 4: Run the new tests and verify they fail for missing implementation**

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q
```

Expected:

```text
FAILED ... Unsupported model preset 'cmass_lens_only'
```

- [ ] **Step 5: Commit the failing contract tests**

```bash
git add Bayesian_inference/tests/conftest.py Bayesian_inference/tests/test_cmass_lens_only_model.py
git commit -m "test: define cmass lens-only model contract"
```

---

### Task 2: Add ModelSpec, Context, Runtime, and Registry Entry

**Files:**
- Create: `Bayesian_inference/src/cmass_lens_inference/components/population/stellar_mass_function/gaussian_lens_sample.py`
- Modify: `Bayesian_inference/src/cmass_lens_inference/components/population/stellar_mass_function/__init__.py`
- Create: `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/__init__.py`
- Create: `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/assembly.py`
- Create: `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/context.py`
- Create: `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/preprocessing.py`
- Create: `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/runtime.py`
- Modify: `Bayesian_inference/src/cmass_lens_inference/model_registry.py`

- [ ] **Step 1: Create the observed lens-sample stellar-mass component**

Create `Bayesian_inference/src/cmass_lens_inference/components/population/stellar_mass_function/gaussian_lens_sample.py`:

```python
"""Gaussian stellar-mass component for an already-observed lens sample."""

from __future__ import annotations

from collections.abc import Mapping

from ....model_interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def gaussian_lens_sample_stellar_mass_component(
    *,
    parameters: tuple[ParameterSpec, ...],
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """
    Return the lens-only Gaussian stellar-mass distribution declaration.

    This is not the parent stellar-mass function used by selection-corrected
    CMASS.  It describes the distribution of stellar masses inside the
    already-observed lens sample, matching the Sonnenfeld-style lens-only
    inference target.
    """

    return ComponentSpec(
        name="population.stellar_mass_function.gaussian_lens_sample",
        kind="stellar_mass_function",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("distributions", "normal_pdf"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["gaussian_lens_sample_stellar_mass_component"]
```

Replace `Bayesian_inference/src/cmass_lens_inference/components/population/stellar_mass_function/__init__.py` with:

```python
"""Stellar-mass-function component declarations."""

from .gaussian_lens_sample import gaussian_lens_sample_stellar_mass_component
from .skewnormal import skewnormal_stellar_mass_function_component
from .smooth_truncated_schechter import smooth_truncated_schechter_component

__all__ = [
    "gaussian_lens_sample_stellar_mass_component",
    "skewnormal_stellar_mass_function_component",
    "smooth_truncated_schechter_component",
]
```

- [ ] **Step 2: Create package exports**

Create `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/__init__.py`:

```python
"""CMASS lens-only model package."""

from .assembly import get_model_spec
from .runtime import get_runtime_adapter

__all__ = ["get_model_spec", "get_runtime_adapter"]
```

- [ ] **Step 3: Create the lens-only context wrapper**

Create `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/context.py`:

```python
"""Array context for the CMASS lens-only model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cmass.context import CMASSModelContext


@dataclass(frozen=True)
class CMASSLensOnlyContext:
    """
    Parameter-independent arrays consumed by the CMASS lens-only posterior.

    `base` stores the existing CMASS per-lens grids and deterministic
    covariates.  `mstar_observation_density` stores only the stellar-mass
    measurement likelihood; the lens-only stellar-mass population term depends
    on sampled parameters and is therefore evaluated in the posterior kernel.
    """

    base: CMASSModelContext
    mstar_observation_density: np.ndarray


__all__ = ["CMASSLensOnlyContext"]
```

- [ ] **Step 4: Create the lens-only assembly**

Create `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/assembly.py`:

```python
"""Assembly layer for the CMASS lens-only model."""

from __future__ import annotations

from ...canonical_dataset import (
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from ...components import (
    aggregate_optional_capabilities,
    aggregate_parameters,
    aggregate_required_capabilities,
)
from ...components.observations.lens_sample import lens_sample_component
from ...components.population.aperture_mass_relation.gaussian_linear import (
    gaussian_linear_aperture_mass_component,
)
from ...components.population.gamma_relation.sigma_star_linear import (
    sigma_star_linear_gamma_component,
)
from ...components.population.size_relation.linear import linear_size_relation_component
from ...components.population.stellar_mass_function.gaussian_lens_sample import (
    gaussian_lens_sample_stellar_mass_component,
)
from ...mass_definition import H_UNITS_V1
from ...model_interfaces import ModelSpec, ParameterSpec
from ..cmass.assembly import (
    GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
    GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE,
    MASS_APERTURE_KPC,
    MODEL_COMPONENT_KEY,
)


MODEL_NAME = "cmass_lens_only"
BACKEND_KERNEL = "cmass_lens_only"
LENS_STELLAR_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_mstar_lens",
    "sigma_mstar_lens",
)
INTERNAL_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
)
SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_sigma_star_gamma",
    "sigma_gamma",
)
INTERNAL_PARAMETER_NAMES: tuple[str, ...] = (
    LENS_STELLAR_MASS_PARAMETER_NAMES
    + INTERNAL_MASS_PARAMETER_NAMES
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
)
PUBLIC_PARAMETER_NAMES: tuple[str, ...] = (
    LENS_STELLAR_MASS_PARAMETER_NAMES
    + ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
)
DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu_mstar_lens": (10.0, 12.5),
    "sigma_mstar_lens": (1.0e-3, 1.0),
    "mu5_0": (9.0, 12.0),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.2),
    "mu_gamma_0": (1.5, 2.5),
    "beta_sigma_star_gamma": (-3.0, 3.0),
    "sigma_gamma": (1.0e-3, 0.5),
}
PARAMETER_SPECS: tuple[ParameterSpec, ...] = tuple(
    ParameterSpec(
        internal_name=internal_name,
        public_name=public_name,
        bounds=DEFAULT_BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME[internal_name],
    )
    for internal_name, public_name in zip(
        INTERNAL_PARAMETER_NAMES,
        PUBLIC_PARAMETER_NAMES,
        strict=True,
    )
)
PARAMETER_SPEC_BY_INTERNAL_NAME: dict[str, ParameterSpec] = {
    parameter.internal_name: parameter for parameter in PARAMETER_SPECS
}

COMPONENTS = (
    lens_sample_component(
        required_context_fields=(
            "log_mstar_obs",
            "log_mstar_err",
            "log_re_obs",
            "n_obs",
            "num_sigma",
            "sigma_obs",
            "sigma_err",
        ),
        required_capabilities=(CAPABILITY_LENS_OBSERVATIONS_V1,),
    ),
    gaussian_lens_sample_stellar_mass_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in LENS_STELLAR_MASS_PARAMETER_NAMES
        ),
        required_context_fields=("mstar_grid", "mstar_observation_density"),
    ),
    linear_size_relation_component(
        required_context_fields=("delta_r_grid",),
    ),
    gaussian_linear_aperture_mass_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name] for name in INTERNAL_MASS_PARAMETER_NAMES
        ),
        required_context_fields=(
            "mstar_shift11p4",
            "delta_r_grid",
            "mass_grid_int",
            "dmass_dthetaein_grid_int",
        ),
        required_capabilities=(CAPABILITY_LENSING_MASS_GRIDS_V1,),
    ),
    sigma_star_linear_gamma_component(
        parameters=tuple(
            PARAMETER_SPEC_BY_INTERNAL_NAME[name]
            for name in SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
        ),
        required_context_fields=(
            "gamma_grid_int",
            "sigma_star_shift9p0_grid",
        ),
    ),
)

PARAMETERS = aggregate_parameters(COMPONENTS)
if PARAMETERS != PARAMETER_SPECS:
    raise ValueError("CMASS lens-only component parameter blocks do not match the schema.")


def get_model_spec() -> ModelSpec:
    """
    Return the scientific specification for the CMASS lens-only model.

    This model follows the Sonnenfeld-style lens-only comparison: it fits the
    distribution of the already-observed lens sample and intentionally excludes
    source-redshift population parameters, discovery-probability parameters,
    lensing cross-section weights, and selection normalization.
    """

    return ModelSpec(
        name=MODEL_NAME,
        component_key=MODEL_COMPONENT_KEY,
        required_unit_convention=H_UNITS_V1,
        mass_aperture_kpc=MASS_APERTURE_KPC,
        parameters=PARAMETERS,
        metadata={
            "component_assembly": (
                "lens_observations -> lens_sample_stellar_mass_distribution -> "
                "linear_size_relation -> enclosed_mass_population -> gamma_population; "
                "posterior_kernel adds observed_sigma_likelihood"
            ),
            "gamma_distribution": GAMMA_DISTRIBUTION_SIGMA_STAR_DEPENDENT,
            "mass_definition": "m5_hinvkpc",
            "unit_convention": H_UNITS_V1,
            "selection_correction": False,
            "fp_prior_supported": False,
            "observed_velocity_dispersion_component": False,
            "target_population": "observed_cmass_lenses",
        },
        required_capabilities=aggregate_required_capabilities(
            COMPONENTS,
            extra=(CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,),
        ),
        optional_capabilities=aggregate_optional_capabilities(COMPONENTS),
        static_codes={"gamma_mode": GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE},
        backend_kernel=BACKEND_KERNEL,
    )


__all__ = [
    "BACKEND_KERNEL",
    "COMPONENTS",
    "INTERNAL_PARAMETER_NAMES",
    "MODEL_NAME",
    "PARAMETER_SPECS",
    "PARAMETERS",
    "PUBLIC_PARAMETER_NAMES",
    "get_model_spec",
]
```

- [ ] **Step 5: Create preprocessing**

Create `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/preprocessing.py`:

```python
"""Deterministic preprocessing for the CMASS lens-only model."""

from __future__ import annotations

import math

import numpy as np

from ...canonical_context import canonical_dataset_metadata
from ...canonical_dataset import (
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CanonicalInferenceDataset,
    load_canonical_inference_dataset,
)
from ...model_interfaces import CompiledContextBundle
from ...profiles import build_profile_spec
from ...types import ProfileSpec, RuntimeConfig
from ..cmass.preprocessing import build_cmass_context_from_canonical_dataset
from .context import CMASSLensOnlyContext


def required_canonical_capabilities(_runtime_config: RuntimeConfig) -> tuple[str, ...]:
    """
    Return canonical capabilities required by the lens-only model.

    The current canonical reader still expects a cross-section block to exist,
    but the lens-only model does not require the cross-section capability and
    the posterior ignores the loaded cross-section values.  The per-lens S2
    velocity-dispersion capability is required because the lens-only posterior
    still evaluates the observed velocity-dispersion likelihood.
    """

    return (
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    )


def load_cmass_lens_only_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    profile: ProfileSpec | None = None,
) -> CanonicalInferenceDataset:
    """Load the canonical dataset with lens-only scientific requirements."""

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("CMASS lens-only preprocessing requires data.inference_dataset_path.")

    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    return load_canonical_inference_dataset(
        runtime_config.data.inference_dataset_path,
        expected_unit_convention=runtime_config.unit_convention,
        expected_h_ref=runtime_config.h_ref,
        expected_profile_name=active_profile.name,
        expected_mass_definition_label=runtime_config.mass_definition.label,
        required_capabilities=required_canonical_capabilities(runtime_config),
    )


def _mstar_observation_density(
    *,
    dataset: CanonicalInferenceDataset,
    mstar_grid: np.ndarray,
) -> np.ndarray:
    """
    Build P(logMstar_obs | logMstar) on each per-lens quadrature grid.

    This is separated from the lens stellar-mass population density because
    the latter depends on sampled parameters `mu_mstar_lens` and
    `sigma_mstar_lens`.
    """

    sqrt2pi = math.sqrt(2.0 * math.pi)
    density = np.zeros_like(mstar_grid, dtype=np.float64)
    for lens_index in range(mstar_grid.shape[0]):
        observed = float(dataset.lenses.log_mstar_obs[lens_index])
        error = float(dataset.lenses.log_mstar_err[lens_index])
        if error <= 0.0:
            raise ValueError(
                "CMASS lens-only requires positive log_mstar_err for every lens; "
                f"lens index {lens_index} has {error}."
            )
        values = mstar_grid[lens_index]
        density[lens_index] = np.exp(-0.5 * ((observed - values) / error) ** 2)
        density[lens_index] /= error * sqrt2pi
    return np.ascontiguousarray(density, dtype=np.float64)


def build_cmass_lens_only_context_from_canonical_dataset(
    runtime_config: RuntimeConfig,
    *,
    dataset: CanonicalInferenceDataset | None = None,
    profile: ProfileSpec | None = None,
) -> CompiledContextBundle:
    """
    Build the CMASS lens-only context from a canonical inference dataset.

    The base CMASS context is reused for mass-grid interpolation, kinematic
    grids, h-unit pivots, and per-lens deterministic covariates.  The posterior
    only consumes the subset relevant to a lens-only likelihood.
    """

    if runtime_config.fp_prior.enabled:
        raise ValueError("cmass_lens_only does not support fp_prior.enabled=true.")

    active_profile = profile or build_profile_spec(runtime_config.profile.name)
    active_dataset = dataset or load_cmass_lens_only_canonical_dataset(
        runtime_config,
        profile=active_profile,
    )
    base_bundle = build_cmass_context_from_canonical_dataset(
        runtime_config,
        dataset=active_dataset,
        profile=active_profile,
    )
    base_context = base_bundle.context
    lens_only_context = CMASSLensOnlyContext(
        base=base_context,
        mstar_observation_density=_mstar_observation_density(
            dataset=active_dataset,
            mstar_grid=base_context.mstar_grid,
        ),
    )
    metadata = {
        **canonical_dataset_metadata(active_dataset),
        "selection_correction": False,
        "target_population": "observed_cmass_lenses",
    }
    return CompiledContextBundle(
        context=lens_only_context,
        profile=base_bundle.profile,
        cross_section_grid=base_bundle.cross_section_grid,
        cosmology=base_bundle.cosmology,
        random_basis=base_bundle.random_basis,
        observations=(),
        metadata=metadata,
    )


__all__ = [
    "build_cmass_lens_only_context_from_canonical_dataset",
    "load_cmass_lens_only_canonical_dataset",
    "required_canonical_capabilities",
]
```

- [ ] **Step 6: Create runtime adapter**

Create `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/runtime.py`:

```python
"""Runtime adapter for the CMASS lens-only model."""

from __future__ import annotations

from ...model_interfaces import (
    CompiledContextBundle,
    ContextArraySpec,
    DataSpec,
    ModelRuntimeAdapter,
)
from ...types import RuntimeConfig
from .preprocessing import build_cmass_lens_only_context_from_canonical_dataset


def build_context_bundle(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """Build the lens-only source-context bundle for the generic backend."""

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("The CMASS lens-only runtime requires data.inference_dataset_path.")
    return build_cmass_lens_only_context_from_canonical_dataset(runtime_config)


def get_data_spec() -> DataSpec:
    """
    Return the lens-only context declaration.

    The current production backend stores the source context directly.  This
    declaration still names the model-owned array that is unique to lens-only
    evaluation so future packed backends have an explicit contract.
    """

    return DataSpec(
        backend_context_type=object,
        array_fields=(ContextArraySpec("mstar_observation_density"),),
        scalar_fields=(),
        static_fields=(),
        normalization_samples_field="base.base_normals",
        normalization_min_value_field="base.normalization_min_value",
    )


def get_runtime_adapter() -> ModelRuntimeAdapter:
    """Return the runtime adapter paired with `cmass_lens_only.get_model_spec()`."""

    return ModelRuntimeAdapter(
        build_context_bundle=build_context_bundle,
        data_spec=get_data_spec(),
    )


__all__ = ["build_context_bundle", "get_data_spec", "get_runtime_adapter"]
```

- [ ] **Step 7: Register model with a temporary posterior import target**

Modify `Bayesian_inference/src/cmass_lens_inference/model_registry.py`:

```python
from .models import (
    cmass,
    cmass_runtime,
    cmass_lens_only,
    cmass_lens_only_runtime,
    sonnenfeld2024_slacs,
    ...
)
from .models.cmass_lens_only import posterior as cmass_lens_only_posterior
```

Add the registry branch before `sonnenfeld2024_slacs`:

```python
    if model_name == "cmass_lens_only":
        return build_model_definition(
            cmass_lens_only.get_model_spec(),
            cmass_lens_only_runtime.get_runtime_adapter(),
            cmass_lens_only_posterior.log_prob,
        )
```

Extend the unsupported-model message to include `cmass_lens_only`.

- [ ] **Step 8: Run registry tests**

Run:

```bash
conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_cmass_lens_only_model.py::test_cmass_lens_only_is_registered_as_concrete_model \
  Bayesian_inference/tests/test_cmass_lens_only_model.py::test_cmass_lens_only_config_has_lens_only_parameter_schema \
  -q
```

Expected before Task 3:

```text
ERROR ... cannot import name 'posterior' from cmass_lens_only
```

- [ ] **Step 9: Commit model shell**

```bash
git add \
  Bayesian_inference/src/cmass_lens_inference/components/population/stellar_mass_function \
  Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only \
  Bayesian_inference/src/cmass_lens_inference/model_registry.py
git commit -m "feat: add cmass lens-only model shell"
```

---

### Task 3: Implement the Lens-Only Posterior Kernel

**Files:**
- Create: `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/posterior.py`

- [ ] **Step 1: Create posterior with explicit theta unpacking**

Create `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/posterior.py`:

```python
"""CMASS lens-only posterior and model-owned Numba kernels."""

from __future__ import annotations

import math
from time import perf_counter

import numba as nb
import numpy as np

from ...numba_backend.diagnostics import build_timing_blob
from ...numba_backend.kernels.distributions import normal_pdf
from ...numba_backend.kernels.integration import trapezoid_1d
from ...numba_backend.kernels.population import (
    gaussian_linear_mass_mean,
    sigma_star_linear_gamma_mean,
)
from ...numba_backend.kernels.selection_likelihood import (
    observed_sigma_likelihood,
    sigma_model_from_s2,
)
from ...types import CompiledModel
from .assembly import GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE


THETA_DIMENSION = 9


@nb.njit(cache=True, inline="always")
def unpack_lens_only_theta(theta: np.ndarray) -> tuple[float, ...]:
    """Return the fixed scalar tuple used by the CMASS lens-only posterior."""

    return (
        theta[0],
        theta[1],
        theta[2],
        theta[3],
        theta[4],
        theta[5],
        theta[6],
        theta[7],
        theta[8],
    )
```

- [ ] **Step 2: Add the lens-only likelihood kernel**

Append this kernel to `posterior.py`:

```python
@nb.njit(cache=True, parallel=True, fastmath=True)
def log_likelihood_lenses_only_numba(
    theta: np.ndarray,
    mass_grid_int: np.ndarray,
    dmass_dthetaein_grid_int: np.ndarray,
    s2_grid_int: np.ndarray,
    has_s2: np.ndarray,
    num_sigma: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    mstar_grid: np.ndarray,
    mstar_shift11p4: np.ndarray,
    sigma_star_shift9p0_grid: np.ndarray,
    mstar_observation_density: np.ndarray,
    delta_r_grid: np.ndarray,
    gamma_grid_int: np.ndarray,
    gamma_mode_code: int,
) -> float:
    """
    Evaluate the observed-lens-sample likelihood.

    This kernel deliberately excludes every selection-correction term present
    in the default CMASS model: no source-redshift density, no lensing
    cross-section, no lens-finding probability, and no selection normalization.
    """

    if theta.shape[0] != THETA_DIMENSION:
        return -np.inf

    (
        mu_mstar_lens,
        sigma_mstar_lens,
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_sigma_star_gamma,
        sigma_gamma,
    ) = unpack_lens_only_theta(theta)

    if sigma_mstar_lens <= 0.0 or sigma5 <= 0.0 or sigma_gamma <= 0.0:
        return -np.inf
    if gamma_mode_code != GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return -np.inf

    n_lens = mstar_grid.shape[0]
    n_gamma = gamma_grid_int.shape[0]
    n_mstar = mstar_grid.shape[1]
    ll_terms = np.zeros(n_lens, dtype=np.float64)
    valid = np.ones(n_lens, dtype=np.int64)

    for lens_index in nb.prange(n_lens):
        gamma_integrand = np.zeros(n_gamma, dtype=np.float64)
        for gamma_index in range(n_gamma):
            gamma = gamma_grid_int[gamma_index]
            log_enclosed_mass = mass_grid_int[lens_index, gamma_index]
            jacobian = abs(dmass_dthetaein_grid_int[lens_index, gamma_index])
            if jacobian <= 0.0:
                continue

            sigma_model = sigma_model_from_s2(
                s2_grid_int[lens_index, gamma_index],
                log_enclosed_mass,
            )
            sigma_probability = observed_sigma_likelihood(
                lens_index,
                num_sigma,
                has_s2,
                sigma_obs,
                sigma_err,
                sigma_model,
            )
            if sigma_probability <= 0.0:
                continue

            mstar_integrand = np.zeros(n_mstar, dtype=np.float64)
            for mstar_index in range(n_mstar):
                mstar_value = mstar_grid[lens_index, mstar_index]
                mstar_obs_density = mstar_observation_density[lens_index, mstar_index]
                if mstar_obs_density <= 0.0:
                    continue

                mstar_lens_density = normal_pdf(
                    mstar_value,
                    mu_mstar_lens,
                    sigma_mstar_lens,
                )
                if mstar_lens_density <= 0.0:
                    continue

                mu5 = gaussian_linear_mass_mean(
                    mu5_0,
                    beta5,
                    xi5,
                    mstar_shift11p4[lens_index, mstar_index],
                    delta_r_grid[lens_index, mstar_index],
                )
                mu_gamma = sigma_star_linear_gamma_mean(
                    mu_gamma_0,
                    beta_sigma_star_gamma,
                    sigma_star_shift9p0_grid[lens_index, mstar_index],
                )
                mstar_integrand[mstar_index] = (
                    mstar_obs_density
                    * mstar_lens_density
                    * normal_pdf(log_enclosed_mass, mu5, sigma5)
                    * normal_pdf(gamma, mu_gamma, sigma_gamma)
                )

            integrated_mstar = trapezoid_1d(mstar_integrand, mstar_grid[lens_index])
            gamma_integrand[gamma_index] = integrated_mstar * jacobian * sigma_probability

        lens_integral = trapezoid_1d(gamma_integrand, gamma_grid_int)
        if lens_integral <= 0.0 or not math.isfinite(lens_integral):
            valid[lens_index] = 0
            continue
        ll_terms[lens_index] = math.log(lens_integral)

    total = 0.0
    for lens_index in range(n_lens):
        if valid[lens_index] == 0:
            return -np.inf
        total += ll_terms[lens_index]
    return total
```

- [ ] **Step 3: Add emcee-compatible `log_prob`**

Append:

```python
def log_prob(theta: np.ndarray, compiled_model: CompiledModel, total_start: float) -> tuple[float, np.void]:
    """
    Evaluate the CMASS lens-only posterior.

    Diagnostic fields retain the shared backend schema. `normalization_value`
    is set to 1.0 because this model has no selection-normalization integral.
    """

    lens_only_context = compiled_model.context
    base = lens_only_context.base

    likelihood_start = perf_counter()
    likelihood_value = log_likelihood_lenses_only_numba(
        theta=theta,
        mass_grid_int=base.mass_grid_int,
        dmass_dthetaein_grid_int=base.dmass_dthetaein_grid_int,
        s2_grid_int=base.s2_grid_int,
        has_s2=base.has_s2,
        num_sigma=base.num_sigma,
        sigma_obs=base.sigma_obs,
        sigma_err=base.sigma_err,
        mstar_grid=base.mstar_grid,
        mstar_shift11p4=base.mstar_shift11p4,
        sigma_star_shift9p0_grid=base.sigma_star_shift9p0_grid,
        mstar_observation_density=lens_only_context.mstar_observation_density,
        delta_r_grid=base.delta_r_grid,
        gamma_grid_int=base.gamma_grid_int,
        gamma_mode_code=base.gamma_mode_code,
    )
    likelihood_seconds = perf_counter() - likelihood_start
    total_seconds = perf_counter() - total_start
    blob = build_timing_blob(
        total_log_prob_seconds=total_seconds,
        likelihood_seconds=likelihood_seconds,
        normalization_seconds=0.0,
        fp_prior_seconds=0.0,
        normalization_value=1.0,
        fp_prior_log_term=0.0,
        fpfit_mu=math.nan,
        fpfit_beta=math.nan,
        fpfit_xi=math.nan,
        fpfit_scatter=math.nan,
        kernel="cmass_lens_only",
        parallel_strategy=compiled_model.parallelism.strategy,
    )

    if not np.isfinite(likelihood_value):
        return -np.inf, blob
    return float(likelihood_value), blob


__all__ = ["log_prob", "log_likelihood_lenses_only_numba", "unpack_lens_only_theta"]
```

- [ ] **Step 4: Run contract tests**

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q
```

Expected after this task:

```text
3 passed
```

- [ ] **Step 5: Commit posterior implementation**

```bash
git add Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/posterior.py
git commit -m "feat: implement cmass lens-only posterior"
```

---

### Task 4: Prove the Model Ignores Selection and Cross-Section Weights

**Files:**
- Modify: `Bayesian_inference/tests/test_cmass_lens_only_model.py`

- [ ] **Step 1: Add finite log-prob test**

Append:

```python
def test_cmass_lens_only_numba_log_prob_is_finite(
    synthetic_lens_only_config_path: Path,
) -> None:
    """A valid lens-only initial point should produce a finite posterior."""

    runtime_config = load_runtime_config(synthetic_lens_only_config_path)
    theta = runtime_config.sampling.initial_center.to_array()
    numba_model = build_numba_model(runtime_config)

    value, blob = numba_log_prob(theta, numba_model)

    assert np.isfinite(value)
    assert blob["kernel"].decode("utf-8").rstrip("\x00") == "cmass_lens_only"
    assert float(blob["normalization_value"]) == pytest.approx(1.0)
    assert float(blob["fp_prior_log_term"]) == pytest.approx(0.0)
```

- [ ] **Step 2: Add cross-section invariance test**

Append:

```python
def test_cmass_lens_only_log_prob_is_independent_of_cross_section_grid(
    synthetic_lens_only_config_path: Path,
    tmp_path: Path,
) -> None:
    """Changing cross-section values should not change lens-only likelihood."""

    payload = yaml.safe_load(synthetic_lens_only_config_path.read_text(encoding="utf-8"))
    original_dataset_path = Path(payload["data"]["inference_dataset_path"]).resolve()
    altered_dataset_path = tmp_path / "altered_cross_section.hdf5"

    with h5py.File(original_dataset_path, "r") as source, h5py.File(altered_dataset_path, "w") as target:
        source.copy("/", target)
        grid = target["lensing_cross_section"]["cross_section_grid"]
        grid[...] = grid[...] * 1.0e9 + 123.0

    altered_payload = dict(payload)
    altered_payload["data"] = {"inference_dataset_path": str(altered_dataset_path)}
    altered_config_path = tmp_path / "altered_cross_section_config.yaml"
    altered_config_path.write_text(yaml.safe_dump(altered_payload, sort_keys=False), encoding="utf-8")

    original_config = load_runtime_config(synthetic_lens_only_config_path)
    altered_config = load_runtime_config(altered_config_path)
    theta = original_config.sampling.initial_center.to_array()

    original_value, _ = numba_log_prob(theta, build_numba_model(original_config))
    altered_value, _ = numba_log_prob(theta, build_numba_model(altered_config))

    assert np.isfinite(original_value)
    assert altered_value == pytest.approx(original_value, rel=0.0, abs=1.0e-10)
```

- [ ] **Step 3: Add rejection test for source/selection parameters in config**

Append:

```python
def test_cmass_lens_only_rejects_selection_parameters_in_box_prior(
    synthetic_lens_only_config_path: Path,
) -> None:
    """Lens-only configs should not accept removed source/discovery parameters."""

    payload = yaml.safe_load(synthetic_lens_only_config_path.read_text(encoding="utf-8"))
    payload["box_prior"]["theta0"] = [0.0, 3.0]
    payload["sampling"]["initial_center"]["theta0"] = 0.93
    config_path = synthetic_lens_only_config_path.parent / "lens_only_with_theta0.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(KeyError, match="theta0"):
        load_runtime_config(config_path)
```

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit selection-invariance tests**

```bash
git add Bayesian_inference/tests/test_cmass_lens_only_model.py
git commit -m "test: prove cmass lens-only ignores selection weights"
```

---

### Task 5: Add Production Config

**Files:**
- Create: `Bayesian_inference/configs/cmass_lens_only.yaml`

- [ ] **Step 1: Create production config**

Create `Bayesian_inference/configs/cmass_lens_only.yaml`:

```yaml
profile:
  name: sersic

unit_convention: h_units_v1

model:
  name: cmass_lens_only

data:
  inference_dataset_path: /Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sersic_slit_m5_hunits_v1.hdf5

box_prior:
  mu_mstar_lens: [10.0, 12.5]
  sigma_mstar_lens: [1.0e-3, 1.0]
  mu5h_0: [9.0, 12.0]
  beta5h: [-3.0, 3.0]
  xi5h: [-3.0, 3.0]
  sigma5h: [1.0e-2, 0.2]
  mu_gamma_0: [1.5, 2.5]
  beta_sigma_star_gamma: [-3.0, 3.0]
  sigma_gamma: [1.0e-3, 0.5]

sampling:
  n_walkers: 24
  n_steps: 10000
  burn_in: 2000
  random_seed: 7
  initial_jitter_scale: 1.0e-3
  initial_center:
    mu_mstar_lens: 11.1
    sigma_mstar_lens: 0.15
    mu5h_0: 11.32
    beta5h: 0.59
    xi5h: -0.11
    sigma5h: 0.06
    mu_gamma_0: 1.99
    beta_sigma_star_gamma: 0.10
    sigma_gamma: 0.149

integration:
  gamma_points: 200
  mstar_points: 200
  normalization_samples: 100000

cosmology:
  h0: 70.0
  omega_m: 0.3

runtime:
  checkpoint_every: 100
  parallel_strategy: auto
  progress: true
  progress_summary_every: 25
  show_stage_timing: true
  disable_hdf5_file_locking: false
  num_threads: 0
  reserve_cores: 2

output:
  root_dir: /Users/liurongfu/Work/CMASS_lens_project/outputs
  run_label: cmass-lens-only
  overwrite_latest: true
```

- [ ] **Step 2: Parse config**

Run:

```bash
conda run -n cmass_lens python - <<'PY'
from cmass_lens_inference.config import load_runtime_config
cfg = load_runtime_config("Bayesian_inference/configs/cmass_lens_only.yaml")
print(cfg.model.name)
print(cfg.parameter_schema.public_parameter_names)
PY
```

Expected:

```text
cmass_lens_only
('mu_mstar_lens', 'sigma_mstar_lens', 'mu5h_0', 'beta5h', 'xi5h', 'sigma5h', 'mu_gamma_0', 'beta_sigma_star_gamma', 'sigma_gamma')
```

- [ ] **Step 3: Commit config**

```bash
git add Bayesian_inference/configs/cmass_lens_only.yaml
git commit -m "config: add cmass lens-only inference preset"
```

---

### Task 6: Add Documentation Near the Model

**Files:**
- Create: `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/README.md`
- Modify: `Bayesian_inference/docs/model_refactor_progress.md`

- [ ] **Step 1: Add model README**

Create `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/README.md`:

```markdown
# CMASS Lens-Only Model

`cmass_lens_only` is a Sonnenfeld-style lens-only comparison model for CMASS.
It fits the already-observed CMASS lens sample directly.

## Scientific Meaning

The model estimates the distribution of `m5` and `gamma` for observed lenses.
It does not infer a parent population that is later filtered through lensing
selection. This makes its posterior comparable to the "Lens-only" column in
Sonnenfeld 2024 Table 2, not to the fiducial selection-corrected model.

## Included Terms

- observed stellar-mass likelihood
- Gaussian stellar-mass distribution for the observed lens sample
- CMASS h-unit enclosed-mass relation
- CMASS sigma-star-dependent gamma relation
- per-lens Einstein-radius mass grid and Jacobian
- observed velocity-dispersion likelihood, assembled directly in the posterior

## Excluded Terms

- lensing cross-section
- lens-finding probability
- selection normalization
- source-redshift population parameters
- FP prior
- standalone observed-velocity-dispersion likelihood component

## Parameter Order

1. `mu_mstar_lens`
2. `sigma_mstar_lens`
3. `mu5h_0`
4. `beta5h`
5. `xi5h`
6. `sigma5h`
7. `mu_gamma_0`
8. `beta_sigma_star_gamma`
9. `sigma_gamma`

## Implementation Boundary

The model reuses CMASS canonical preprocessing for h-unit pivots, per-lens mass
grids, and velocity-dispersion grids. The posterior has a separate Numba kernel
so selection terms cannot accidentally leak from the default `cmass` posterior.
The posterior imports shared kernels directly instead of depending on helper
functions from `models.cmass.posterior`.
```

- [ ] **Step 2: Update refactor progress doc**

Append a short section to `Bayesian_inference/docs/model_refactor_progress.md`:

```markdown
## CMASS Lens-Only Model

`cmass_lens_only` is implemented as a separate concrete registry model rather
than a configuration switch inside `cmass`. This preserves the scientific
meaning of the default CMASS model as selection-corrected while providing a
Sonnenfeld-style lens-only comparison path.

The model removes source-redshift, discovery-probability, cross-section,
selection-normalization, and FP-prior terms. Its target population is the
observed CMASS lens sample.
```

- [ ] **Step 3: Commit docs**

```bash
git add Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/README.md Bayesian_inference/docs/model_refactor_progress.md
git commit -m "docs: document cmass lens-only semantics"
```

---

### Task 7: End-to-End Verification

**Files:**
- Modify only if tests expose a real bug in previous tasks.

- [ ] **Step 1: Run targeted test file**

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 2: Run existing registry and backend tests**

Run:

```bash
conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_model_registry_config.py \
  Bayesian_inference/tests/test_numba_emcee_inference.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 3: Run full Bayesian inference test suite**

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q
```

Expected:

```text
all non-data-skipped tests pass
```

If real-data tests skip because local HDF5 products are absent, record the exact skip lines in the final handoff.

- [ ] **Step 4: Run a short real-config smoke evaluation without long sampling**

Run:

```bash
conda run -n cmass_lens python - <<'PY'
from dataclasses import replace
from pathlib import Path
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.numba_backend.likelihood_engine import build_compiled_model, log_prob

config_path = Path("Bayesian_inference/configs/cmass_lens_only.yaml")
cfg = load_runtime_config(config_path)
model = build_compiled_model(cfg)
value, blob = log_prob(cfg.sampling.initial_center.to_array(), model)
print("log_prob", value)
print("kernel", blob["kernel"].decode("utf-8").rstrip("\x00"))
print("normalization_value", float(blob["normalization_value"]))
PY
```

Expected:

```text
kernel cmass_lens_only
normalization_value 1.0
```

`log_prob` must be finite. If it is `-inf`, inspect per-lens integrals before changing priors.

- [ ] **Step 5: Run formatting guard**

Run:

```bash
git diff --check
```

Expected:

```text
no output
```

- [ ] **Step 6: Commit verification-only fixes if needed**

If previous steps required small fixes:

```bash
git add Bayesian_inference/src/cmass_lens_inference Bayesian_inference/tests Bayesian_inference/configs Bayesian_inference/docs
git commit -m "fix: stabilize cmass lens-only verification"
```

Skip this commit when no files changed.

---

## Final Acceptance Criteria

The work is complete only when all of the following are true:

- `model.name: cmass_lens_only` parses through the standard config loader.
- `cmass_lens_only` appears in `model_registry.get_model_definition`.
- The registry declares `lens_observations.v1`, `lensing_mass_grids.v1`, and `velocity_dispersion.per_lens_s2.v1` as required capabilities.
- The registry does not declare `lensing_cross_section.theta_gamma_grid.v1` as a required capability.
- The parameter schema has 9 dimensions.
- `mu_zs`, `sigma_zs`, `theta0`, and `loga` are absent from the public parameter order.
- The assembly uses the observed lens-sample Gaussian stellar-mass component, existing size/mass/gamma components, and no standalone observed-velocity-dispersion likelihood component.
- `fp_prior.enabled: true` raises a direct model-specific error.
- A finite synthetic Numba log-prob is produced.
- Diagnostic blob has `kernel == "cmass_lens_only"` and `normalization_value == 1.0`.
- Multiplying the canonical cross-section grid by a huge factor leaves the log-prob unchanged.
- The posterior imports reusable shared kernels directly and does not import helper functions from `models.cmass.posterior`.
- Existing `cmass` tests still pass.
- Full `Bayesian_inference/tests` passes except documented skips caused by missing real datasets.
- The code comments and docstrings explain why the model is lens-only and why selection terms are absent.

## Risks and Guardrails

- **Risk:** Accidentally retaining parent stellar-mass or size likelihood through `mstar_integrand_base`.
  - Guardrail: lens-only posterior must use `mstar_observation_density` and sampled `P_lens(logMstar)`, not `base.mstar_integrand_base`.
- **Risk:** Accidentally retaining cross-section through a reused CMASS helper.
  - Guardrail: cross-section invariance test must pass.
- **Risk:** Calling it "no selection" while still keeping source-redshift parameters.
  - Guardrail: schema test forbids `mu_zs` and `sigma_zs`.
- **Risk:** FP-prior semantics become ambiguous.
  - Guardrail: reject `fp_prior.enabled` in `cmass_lens_only` v1.
- **Risk:** Over-componentizing the hot likelihood path by adding a standalone observed-velocity-dispersion likelihood component.
  - Guardrail: keep `sigma_model_from_s2` and `observed_sigma_likelihood` calls inside the dedicated posterior kernel, while documenting that they are posterior-owned likelihood assembly.
- **Risk:** Coupling lens-only to default CMASS posterior internals.
  - Guardrail: import reusable shared kernels such as `gaussian_linear_mass_mean` and `sigma_star_linear_gamma_mean` directly.
- **Risk:** Data-preparation work expands scope.
  - Guardrail: do not change `prepare_dataset/`; the current canonical reader still loads the cross-section block, but the lens-only posterior ignores it.
