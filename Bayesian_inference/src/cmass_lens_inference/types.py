"""
Typed project models.

This module centralizes all cross-module data structures so the scientific
pipeline remains explicit and inspectable. The dataclasses are intentionally
verbose because the project requirements emphasize maintainability and
handoff-readiness over terse implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mass_definition import MassDefinition
from .parameter_schema import GammaModelConfig, ParameterSchema


def _serialize_json_friendly(value: Any) -> Any:
    """
    Recursively convert project dataclasses into JSON-friendly payloads.

    Why this helper exists:
    - result objects now nest other dataclasses such as latest-summary wrappers
    - plain `asdict()` leaves `Path` instances untouched, which would break the
      CLI's `json.dumps(...)` contract
    - centralizing the conversion keeps every result type consistent instead of
      repeating slightly different recursive serializers
    """

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serialize_json_friendly(field_value) for key, field_value in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize_json_friendly(field_value) for key, field_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_friendly(item) for item in value]
    return value


@dataclass(frozen=True)
class HyperParams:
    """
    Mode-aware sampled hyper-parameters for one run configuration.

    The values are stored as an ordered tuple rather than as fixed dataclass
    fields because the gamma mode now changes the parameter dimension and
    public naming surface.
    """

    parameter_schema: ParameterSchema
    parameter_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.parameter_values) != self.parameter_schema.n_dim:
            raise ValueError(
                "Hyper-parameter tuple length does not match the configured "
                f"schema dimension {self.parameter_schema.n_dim}."
            )

    def to_array(self) -> np.ndarray:
        """Return parameters in the sampler's canonical order."""

        return np.asarray(self.parameter_values, dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray, parameter_schema: ParameterSchema) -> "HyperParams":
        """Construct the dataclass from an ordered parameter vector."""

        parameter_schema.validate_theta_shape(np.asarray(values, dtype=float))
        return cls(
            parameter_schema=parameter_schema,
            parameter_values=tuple(np.asarray(values, dtype=float).tolist()),
        )

    @classmethod
    def from_public_dict(
        cls,
        public_values: dict[str, float],
        parameter_schema: ParameterSchema,
    ) -> "HyperParams":
        """Construct hyper-parameters from the public config naming surface."""

        normalized = parameter_schema.normalize_public_values(public_values)
        return cls(
            parameter_schema=parameter_schema,
            parameter_values=tuple(
                float(normalized[name])
                for name in parameter_schema.internal_parameter_names
            ),
        )

    def to_dict(self) -> dict[str, float]:
        """Return internal-name keyed values for low-level model consumers."""

        return {
            name: float(value)
            for name, value in zip(
                self.parameter_schema.internal_parameter_names,
                self.parameter_values,
                strict=True,
            )
        }

    def to_public_dict(self, mass_definition: MassDefinition | None = None) -> dict[str, float]:
        """
        Serialize the hyper-parameters using the selected public name family.

        The mass-definition argument stays optional so older call sites can keep
        passing it while the schema itself remains the single source of truth.
        """

        return self.parameter_schema.serialize_public_values(
            self.to_dict(),
            mass_definition=mass_definition,
        )


@dataclass(frozen=True)
class ProfileConfig:
    """The profile section loaded directly from YAML."""

    name: str


@dataclass(frozen=True)
class DataConfig:
    """Input file locations required by the inference pipeline."""

    observation_path: Path
    cross_section_path: Path
    sigma_table_path: Path | None = None


@dataclass(frozen=True)
class FPPriorConfig:
    """
    Optional population-level Fundamental Plane prior configuration.

    The values mirror the legacy SLACS reference implementation, but the prior
    remains disabled by default so existing CMASS inference runs preserve their
    historical behavior unless users opt in explicitly.
    """

    enabled: bool
    fit_mstar_min: float = 11.0
    pivot_mstar: float = 11.3
    fiducial_scatter: float = 0.075
    scatter_error: float = 0.003
    mu_v_prior: float = 2.34548
    mu_v_error: float = 0.00611
    beta_v_prior: float = 0.176
    beta_v_error: float = 0.011


@dataclass(frozen=True)
class ModelConfig:
    """
    Configurable scientific-model assembly for the inference backend.

    The previous implementation encoded most modelling choices as Python
    branches: profile name, gamma mode, FP prior flag, and a fixed selection
    model.  The JAX/NumPyro migration makes that contract explicit so a future
    user can switch to a different published model by editing YAML instead of
    touching numerical kernels.  The component values are registry keys; each
    key names a concrete implementation that the backend can validate and use.
    """

    name: str
    components: dict[str, str]


@dataclass(frozen=True)
class SamplingConfig:
    """
    Controls NumPyro sampling and preserves legacy initialization metadata.

    The legacy `n_walkers`, `n_steps`, and `warmup` fields remain present so old
    configs, tests, and run snapshots can still be read.  New production runs
    should prefer the NumPyro-native fields:
    - `num_chains`
    - `num_samples`
    - `num_warmup`
    - `thinning`
    - `chain_method`
    """

    n_walkers: int
    n_steps: int
    warmup: int
    random_seed: int
    initial_center: HyperParams
    initial_jitter_scale: float
    num_chains: int
    num_samples: int
    num_warmup: int
    thinning: int
    chain_method: str


@dataclass(frozen=True)
class IntegrationConfig:
    """Controls the deterministic quadrature and MC normalization sizes."""

    gamma_points: int
    mstar_points: int
    normalization_samples: int


@dataclass(frozen=True)
class CosmologyConfig:
    """
    Physical cosmology parameters shared by every distance calculation.

    These values describe the model's global background cosmology, so they are
    kept separate from execution-only runtime knobs such as checkpoint cadence
    or thread counts.
    """

    h0: float
    omega_m: float


@dataclass(frozen=True)
class RuntimeOptions:
    """Execution-time controls that are not part of the statistical model."""

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

    unit_convention: str
    h_ref: float
    profile: ProfileConfig
    model: ModelConfig
    mass_definition: MassDefinition
    gamma_model: GammaModelConfig
    parameter_schema: ParameterSchema
    fp_prior: FPPriorConfig
    data: DataConfig
    sampling: SamplingConfig
    integration: IntegrationConfig
    cosmology: CosmologyConfig
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
    log_effective_radius_obs: float | None
    einstein_radius_arcsec: float
    num_sigma: int
    sigma_observed: np.ndarray
    sigma_error: np.ndarray
    gamma_grid_17: np.ndarray
    mass_grid_17: np.ndarray
    dmass_dthetaein_grid_17: np.ndarray
    s2_grid_17: np.ndarray | None


@dataclass(frozen=True)
class CrossSectionGrid:
    """The precomputed one-dimensional cross-section lookup table."""

    gamma_grid: np.ndarray
    cs_over_theta_ein: np.ndarray


@dataclass(frozen=True)
class SigmaUnitTable:
    """
    Normalized sigma-unit interpolation table consumed by the FP prior.

    The table stores `sigma^2 / 10**m_R` on an explicit profile- and
    mass-definition-aware grid. Validation lives in the I/O layer so later
    compiled-context code can assume the table matches the active run.
    """

    profile_name: str
    mass_definition_label: str
    mass_radius_kpc: float
    unit_convention: str
    h_ref: float | None
    units: str
    gamma_axis: np.ndarray
    zd_axis: np.ndarray | None
    log_re_kpc_axis: np.ndarray
    sigma_unit_grid: np.ndarray
    n_axis: np.ndarray | None = None
    sigma_definition: str = "observed_aperture"
    bundle_group_name: str = "slit"
    observation_flavor: str | None = "slit"
    aperture_shape: str = "rectangular"
    aperture_width_arcsec: float | None = 1.6
    aperture_height_arcsec: float | None = 0.9
    aperture_radius_arcsec: float | None = None
    seeing_fwhm_arcsec: float | None = 0.9
    bundle_leaf_path: str = "/"


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
    mass_dense: np.ndarray
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
    Fully array-compiled numerical context consumed by the numerical kernels.

    Every field in this dataclass is intentionally backend-friendly: contiguous
    ndarrays, scalar flags, or plain floats. The goal is to move all
    parameter-independent work out of the hot `log_prob` path so the sampler
    only pays for vectorized kernel execution, not Python object orchestration.
    """

    z_grid: np.ndarray
    chi_kpc_grid: np.ndarray
    cs_gamma_grid: np.ndarray
    cs_over_theta_grid: np.ndarray
    cs_over_theta_int: np.ndarray
    gamma_grid_int: np.ndarray
    mass_grid_int: np.ndarray
    dmass_dthetaein_grid_int: np.ndarray
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
    stellar_mass_pivot: float
    sigma_star_shift9p0_grid: np.ndarray
    mstar_integrand_base: np.ndarray
    delta_r_grid: np.ndarray
    base_normals: np.ndarray
    mass_radius_kpc: float
    mass_log_physical_offset: float
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
    gamma_mode_code: int
    fp_enabled: int
    fp_fit_mstar_min: float
    fp_pivot_mstar: float
    fp_fiducial_scatter: float
    fp_scatter_error: float
    fp_mu_v_prior: float
    fp_mu_v_error: float
    fp_beta_v_prior: float
    fp_beta_v_error: float
    fp_gamma_axis: np.ndarray
    fp_zd_axis: np.ndarray
    fp_log_re_kpc_axis: np.ndarray
    fp_n_axis: np.ndarray
    fp_sigma_unit_grid: np.ndarray
    fp_has_n_axis: int


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
    random_basis: RandomBasis | None
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

        return _serialize_json_friendly(self)


@dataclass
class PosteriorCornerResult:
    """Structured summary for one run-directory corner-plot generation."""

    run_id: str
    profile_name: str
    input_run_dir: Path
    figure_path: Path
    result_path: Path
    status: str
    burn_in_applied: int
    n_posterior_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the nested result into CLI- and JSON-friendly data."""

        return _serialize_json_friendly(self)


@dataclass
class PosteriorCornerLatestResult:
    """Bundle the current latest corner-plot results for both profiles."""

    status: str
    devauc_result: PosteriorCornerResult
    sersic_result: PosteriorCornerResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the nested latest-summary structure for CLI output."""

        return _serialize_json_friendly(self)
