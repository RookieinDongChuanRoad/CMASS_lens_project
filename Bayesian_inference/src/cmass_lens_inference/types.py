"""
Typed project models.

This module centralizes all cross-module data structures so the scientific
pipeline remains explicit and inspectable. The dataclasses are intentionally
verbose because the project requirements emphasize maintainability and
handoff-readiness over terse implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)


@dataclass(frozen=True)
class HyperParams:
    """The 12 population hyper-parameters defined by the requirements."""

    mu5_0: float
    beta5: float
    xi5: float
    sigma5: float
    mu_gamma_0: float
    beta_gamma: float
    xi_gamma: float
    sigma_gamma: float
    mu_zs: float
    sigma_zs: float
    theta0: float
    loga: float

    def to_array(self) -> np.ndarray:
        """Return parameters in the sampler's canonical order."""

        return np.array([getattr(self, name) for name in PARAMETER_NAMES], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "HyperParams":
        """Construct the dataclass from an ordered parameter vector."""

        if values.shape != (len(PARAMETER_NAMES),):
            raise ValueError("Hyper-parameter vector must contain exactly 12 values.")
        return cls(**dict(zip(PARAMETER_NAMES, values.tolist(), strict=True)))

    def to_dict(self) -> dict[str, float]:
        """Return a plain dictionary for serialization and metadata output."""

        return {name: float(getattr(self, name)) for name in PARAMETER_NAMES}


@dataclass(frozen=True)
class ProfileConfig:
    """The profile section loaded directly from YAML."""

    name: str


@dataclass(frozen=True)
class DataConfig:
    """Input file locations required by the inference pipeline."""

    observation_path: Path
    cross_section_path: Path


@dataclass(frozen=True)
class SamplingConfig:
    """Controls the `emcee` sampler and walker initialization."""

    n_walkers: int
    n_steps: int
    warmup: int
    random_seed: int
    initial_center: HyperParams
    initial_jitter_scale: float


@dataclass(frozen=True)
class IntegrationConfig:
    """Controls the deterministic quadrature and MC normalization sizes."""

    gamma_points: int
    mstar_points: int
    normalization_samples: int


@dataclass(frozen=True)
class RuntimeOptions:
    """Execution-time controls that are not part of the statistical model."""

    distance_table_max_z: float
    distance_table_size: int
    checkpoint_every: int
    parallel_strategy: str
    progress: bool
    progress_summary_every: int
    show_stage_timing: bool
    disable_hdf5_file_locking: bool
    num_threads: int
    reserve_cores: int


@dataclass(frozen=True)
class OutputConfig:
    """Defines where a run writes artifacts and how `run_id` is formed."""

    root_dir: Path
    run_label: str
    overwrite_latest: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """The fully parsed project configuration."""

    profile: ProfileConfig
    data: DataConfig
    sampling: SamplingConfig
    integration: IntegrationConfig
    runtime: RuntimeOptions
    output: OutputConfig


@dataclass(frozen=True)
class ProfileSpec:
    """
    Profile-specific constants and compatibility rules.

    Keeping all `devauc`/`sersic` differences here prevents profile conditionals
    from leaking into the likelihood or sampler code.
    """

    name: str
    fixed_n: float | None
    uses_observed_n_in_likelihood: bool
    observation_field_aliases: dict[str, tuple[str, ...]]
    mass_function_loc: float
    mass_function_scale: float
    mass_function_alpha: float
    mu_r0: float
    beta_r: float
    sigma_r: float
    nu_r: float | None
    mu_n0: float | None
    beta_n: float | None
    sigma_n: float | None


@dataclass(frozen=True)
class ObservationRecord:
    """Normalized in-memory representation of a single lens observation."""

    lens_id: str
    z_d: float
    z_s: float
    log_stellar_mass_obs: float
    log_stellar_mass_err: float
    n_observed: float
    effective_radius_arcsec: float
    einstein_radius_arcsec: float
    num_sigma: int
    sigma_observed: np.ndarray
    sigma_error: np.ndarray
    gamma_grid_17: np.ndarray
    m5_grid_17: np.ndarray
    dm5_dthetaein_grid_17: np.ndarray
    s2_grid_17: np.ndarray | None


@dataclass(frozen=True)
class CrossSectionGrid:
    """The precomputed one-dimensional cross-section lookup table."""

    gamma_grid: np.ndarray
    cs_over_theta_ein: np.ndarray


@dataclass(frozen=True)
class PreparedObservation:
    """Per-lens arrays precomputed once before sampling begins."""

    lens_id: str
    z_d: float
    z_s: float
    log_stellar_mass_obs: float
    log_stellar_mass_err: float
    n_observed: float
    effective_radius_arcsec: float
    einstein_radius_arcsec: float
    num_sigma: int
    sigma_observed: np.ndarray
    sigma_error: np.ndarray
    gamma_dense: np.ndarray
    m5_dense: np.ndarray
    jacobian_dense: np.ndarray
    s2_dense: np.ndarray | None
    observed_log_effective_radius_kpc: float


@dataclass(frozen=True)
class RandomBasis:
    """
    Fixed pseudo-random basis used to make MC normalization deterministic.

    The second-stage performance refactor aligns the representation with the
    reference implementation: one contiguous matrix of standard-normal draws is
    reused for every normalization evaluation in a run. Individual kernels map
    specific columns to the variables they need.
    """

    base_normals: np.ndarray


@dataclass(frozen=True)
class CompiledModelContext:
    """
    Fully array-compiled numerical context consumed by the production kernels.

    Every field in this dataclass is intentionally `numba`-friendly:
    contiguous ndarrays, scalar flags, or plain floats. The goal is to move all
    parameter-independent work out of the hot `log_prob` path so the sampler
    only pays for kernel execution, not Python object orchestration.
    """

    z_grid: np.ndarray
    chi_kpc_grid: np.ndarray
    cs_gamma_grid: np.ndarray
    cs_over_theta_grid: np.ndarray
    cs_over_theta_int: np.ndarray
    gamma_grid_int: np.ndarray
    gamma_w: np.ndarray
    m5_grid_int: np.ndarray
    jac_grid_int: np.ndarray
    s2_grid_int: np.ndarray
    has_s2: np.ndarray
    num_sigma: np.ndarray
    sigma_obs: np.ndarray
    sigma_err: np.ndarray
    zd: np.ndarray
    zs: np.ndarray
    p_zd_fixed: np.ndarray
    mstar_grid: np.ndarray
    mstar_shift11p4: np.ndarray
    mstar_base: np.ndarray
    delta_r_grid: np.ndarray
    n_obs_for_model: np.ndarray
    base_normals: np.ndarray
    use_sersic_index: int
    n_fixed: float
    mu_n0: float
    beta_n: float
    sigma_n: float
    mass_function_loc: float
    mass_function_scale: float
    mass_function_alpha: float
    mu_r0: float
    beta_r: float
    sigma_r: float
    nu_r: float
    mu_d: float
    sigma_d: float
    gamma_trunc_low: float
    gamma_trunc_high: float
    normalization_min_value: float


@dataclass(frozen=True)
class CompiledModel:
    """
    High-level container for the production `log_prob` entrypoint.

    The compiled context lives alongside the original typed configuration and
    resolved parallel settings so callers can still record rich metadata
    without reintroducing Python object access into the kernels themselves.
    """

    config: RuntimeConfig
    profile: ProfileSpec
    cross_section_grid: CrossSectionGrid
    cosmology: Any
    parallelism: ResolvedParallelism
    context: CompiledModelContext


@dataclass(frozen=True)
class ResolvedParallelism:
    """The concrete parallel execution plan for a run."""

    strategy: str
    cpu_count: int
    reserve_cores: int
    compute_budget: int
    worker_processes: int
    kernel_threads_per_process: int

    def to_dict(self) -> dict[str, int | str]:
        """Serialize resolved parallel settings for metadata output."""

        return {
            "strategy": self.strategy,
            "cpu_count": self.cpu_count,
            "reserve_cores": self.reserve_cores,
            "compute_budget": self.compute_budget,
            "worker_processes": self.worker_processes,
            "kernel_threads_per_process": self.kernel_threads_per_process,
        }


@dataclass(frozen=True)
class RunLayout:
    """Physical filesystem paths associated with a single inference run."""

    root_dir: Path
    profile_dir: Path
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class RuntimeContext:
    """Objects shared by log-probability evaluation and sampler execution."""

    config: RuntimeConfig
    profile: ProfileSpec
    observations: list[ObservationRecord]
    prepared_observations: list[PreparedObservation]
    cross_section_grid: CrossSectionGrid
    random_basis: RandomBasis
    cosmology: Any
    parallelism: ResolvedParallelism
    compiled_model: CompiledModel | None = None


@dataclass
class RunResult:
    """Structured summary written to disk and returned by the public APIs."""

    run_id: str
    profile_name: str
    run_dir: Path
    status: str
    start_step: int
    completed_steps: int
    acceptance_fraction_mean: float
    config_path: Path | None = None
    input_observation_path: Path | None = None
    output_root_dir: Path | None = None
    checkpoint_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into JSON-friendly data."""

        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload
