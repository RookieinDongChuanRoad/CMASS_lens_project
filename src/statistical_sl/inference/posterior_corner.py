"""
Posterior corner-plot workflow for completed CMASS inference runs.

Why this module exists:
- the standard production pipeline treats posterior corner plots as a required
  post-processing artifact rather than a manual notebook-only step
- public labels must read the stored run configuration instead of hard-coding
  one historical naming surface
- production samples are stored in emcee's `chain.h5` HDFBackend, so this
  reader stays decoupled from retired array/netCDF artifact conventions
"""

from __future__ import annotations

import json
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from statistical_sl.inference.config import load_runtime_config
from statistical_sl.inference.types import PosteriorCornerLatestResult, PosteriorCornerResult
from statistical_sl.core.mass_definition import mass_definition_metadata


DEFAULT_DEVAUC_CORNER_RUN_DIR = Path("workspace/outputs/devauc/latest")
DEFAULT_SERSIC_CORNER_RUN_DIR = Path("workspace/outputs/sersic/latest")
POSTERIOR_CORNER_FIGURE_NAME = "posterior_corner.png"
POSTERIOR_CORNER_RESULT_NAME = "posterior_corner_result.json"
POSTERIOR_CORNER_QUANTILES = [0.16, 0.5, 0.84]
POSTERIOR_CORNER_LEVELS = [0.68, 0.95]
POSTERIOR_CORNER_TITLE_FORMAT = ".2f"

_SHARED_PARAMETER_LABELS: dict[str, str] = {
    "mu_mstar_lens": r"$\mu_{\log M_{\ast,\mathrm{lens}}}$",
    "sigma_mstar_lens": r"$\sigma_{\log M_{\ast,\mathrm{lens}}}$",
    "mu_gamma_0": r"$\mu_{\gamma,0}$",
    "beta_gamma": r"$\beta_{\gamma}$",
    "xi_gamma": r"$\xi_{\gamma}$",
    "beta_sigma_star_gamma": r"$\beta_{\Sigma_\ast,\gamma}$",
    "sigma_gamma": r"$\sigma_{\gamma}$",
    "mu_zs": r"$\mu_{z_{\mathrm{s}}}$",
    "sigma_zs": r"$\sigma_{z_{\mathrm{s}}}$",
    "theta0": r"$\theta_0$",
    "loga": r"$\log_{10} a$",
}


def _resolve_run_dir(run_dir: str | Path) -> Path:
    """Normalize a run-directory input and eagerly resolve `latest` symlinks."""

    return Path(run_dir).expanduser().resolve()


def _resolve_burn_in(requested_burn_in: str | int, warmup: int) -> int:
    """
    Normalize the CLI/API burn-in value into a concrete integer.

    The workflow reuses the same `auto => stored warmup` semantics as the PPC
    commands so operators do not need to remember two different post-run
    discard policies.
    """

    if isinstance(requested_burn_in, str):
        if requested_burn_in != "auto":
            raise ValueError("Burn-in must be an integer or the literal string 'auto'.")
        return int(warmup)
    return int(requested_burn_in)


def _load_flattened_emcee_chain(chain_path: Path, burn_in: int) -> np.ndarray:
    """
    Load and flatten the post-burn-in posterior chain.

    The stored HDF5 backend uses the canonical `(step, walker, parameter)`
    layout from `emcee`. The corner plot expects a 2D sample matrix, so this
    helper keeps the flattening policy explicit and shared by both the single-
    run and latest-two APIs.
    """

    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "Reading production chain.h5 files requires the dependency `emcee`."
        ) from exc

    backend = emcee.backends.HDFBackend(str(chain_path))
    chain = backend.get_chain()
    if burn_in >= chain.shape[0]:
        raise ValueError(
            f"Burn-in {burn_in} removes all samples from chain with {chain.shape[0]} stored steps."
        )
    return chain[burn_in:].reshape(-1, chain.shape[-1])


def _mass_parameter_label_lookup(runtime_config) -> dict[str, str]:
    """
    Return public labels for the active aperture-mass population parameters.

    CMASS-family models expose mass-parameter names through the configured
    mass definition (`mu5h_0`, `mu10_0`, ...), while newer model variants such
    as `cmass_lens_only` are free to place those parameters anywhere in the
    sampled vector.  Building this lookup by public name keeps the labels tied
    to the schema rather than to a historical positional convention.
    """

    mass_definition = runtime_config.mass_definition
    radius_label = int(mass_definition.radius_kpc)
    mass_labels = (
        rf"$\mu_{{{radius_label},0}}$",
        rf"$\beta_{{{radius_label}}}$",
        rf"$\xi_{{{radius_label}}}$",
        rf"$\sigma_{{{radius_label}}}$",
    )
    return dict(zip(mass_definition.public_parameter_names, mass_labels, strict=True))


def _public_parameter_order_and_labels(runtime_config) -> tuple[list[str], list[str]]:
    """
    Return the user-visible parameter order and mathtext labels for one run.

    The sampler stores samples in the active model's schema order.  Older CMASS
    models start with the four aperture-mass parameters, but `cmass_lens_only`
    starts with two observed-lens stellar-mass distribution parameters.  This
    resolver therefore labels parameters by public name rather than by vector
    position, while still deriving the mass labels from the exact run's mass
    definition.
    """

    public_parameter_order = list(runtime_config.parameter_schema.public_parameter_names)
    label_lookup = {
        **_SHARED_PARAMETER_LABELS,
        **_mass_parameter_label_lookup(runtime_config),
    }
    public_parameter_labels = [label_lookup.get(name, name) for name in public_parameter_order]
    return public_parameter_order, public_parameter_labels


def _load_corner_runtime_context(resolved_run_dir: Path, burn_in: str | int) -> tuple[np.ndarray, int, str, object]:
    """
    Load the stored config snapshot plus the flattened posterior samples.

    Returning the parsed runtime config alongside the samples keeps all
    downstream serialization logic grounded in the exact run contract that
    produced the chain.
    """

    config_path = resolved_run_dir / "config_snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Run directory '{resolved_run_dir}' does not contain the required config_snapshot.yaml."
        )

    runtime_config = load_runtime_config(config_path)
    chain_path = resolved_run_dir / "chain.h5"
    if chain_path.exists():
        burn_in_steps = _resolve_burn_in(burn_in, runtime_config.sampling.warmup)
        flattened_chain = _load_flattened_emcee_chain(chain_path=chain_path, burn_in=burn_in_steps)
        return flattened_chain, burn_in_steps, runtime_config.profile.name, runtime_config

    raise FileNotFoundError(
        f"Run directory '{resolved_run_dir}' does not contain production chain.h5."
    )


def _write_corner_figure(
    figure_path: Path,
    posterior_samples: np.ndarray,
    profile_name: str,
    parameter_labels: list[str],
) -> None:
    """
    Render the corner plot using the agreed style contract.

    We intentionally delegate the lower-triangle density and diagonal-quantile
    rendering to the `corner` package because the user explicitly wants that
    visual grammar and it is easy to subtly regress if reimplemented by hand.
    """

    posterior_samples = np.asarray(posterior_samples, dtype=float)
    if posterior_samples.ndim != 2 or posterior_samples.shape[1] != len(parameter_labels):
        raise ValueError(
            "Posterior corner plotting expects a 2D array with one column per "
            "mode-aware parameter."
        )

    figure = corner.corner(
        posterior_samples,
        labels=parameter_labels,
        titles=parameter_labels,
        show_titles=True,
        title_fmt=POSTERIOR_CORNER_TITLE_FORMAT,
        quantiles=POSTERIOR_CORNER_QUANTILES,
        plot_datapoints=False,
        levels=POSTERIOR_CORNER_LEVELS,
    )
    figure.suptitle(f"Posterior Corner Plot: {profile_name}", fontsize=16)
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def run_posterior_corner(run_dir: str | Path, burn_in: str | int = "auto") -> PosteriorCornerResult:
    """
    Generate one posterior corner plot inside a completed run directory.

    The workflow writes both the figure and a JSON summary back into the same
    run directory so the artifact stays coupled to the exact chain that
    produced it.
    """

    resolved_run_dir = _resolve_run_dir(run_dir)
    posterior_samples, burn_in_steps, profile_name, runtime_config = _load_corner_runtime_context(
        resolved_run_dir=resolved_run_dir,
        burn_in=burn_in,
    )
    public_parameter_order, public_parameter_labels = _public_parameter_order_and_labels(runtime_config)

    figure_path = resolved_run_dir / POSTERIOR_CORNER_FIGURE_NAME
    _write_corner_figure(
        figure_path=figure_path,
        posterior_samples=posterior_samples,
        profile_name=profile_name,
        parameter_labels=public_parameter_labels,
    )

    result_path = resolved_run_dir / POSTERIOR_CORNER_RESULT_NAME
    result = PosteriorCornerResult(
        run_id=resolved_run_dir.name,
        profile_name=profile_name,
        input_run_dir=resolved_run_dir,
        figure_path=figure_path,
        result_path=result_path,
        status="completed",
        burn_in_applied=int(burn_in_steps),
        n_posterior_samples=int(posterior_samples.shape[0]),
        metadata={
            "parameter_order": public_parameter_order,
            "parameter_labels": public_parameter_labels,
            "figure_name": POSTERIOR_CORNER_FIGURE_NAME,
            "library": "corner",
            "mass_definition": mass_definition_metadata(runtime_config.mass_definition),
            "style": {
                "show_titles": True,
                "title_fmt": POSTERIOR_CORNER_TITLE_FORMAT,
                "quantiles": POSTERIOR_CORNER_QUANTILES,
                "plot_datapoints": False,
                "levels": POSTERIOR_CORNER_LEVELS,
            },
        },
    )
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_latest_profile_corner_plots(
    devauc_run_dir: str | Path = DEFAULT_DEVAUC_CORNER_RUN_DIR,
    sersic_run_dir: str | Path = DEFAULT_SERSIC_CORNER_RUN_DIR,
    burn_in: str | int = "auto",
) -> PosteriorCornerLatestResult:
    """
    Generate corner plots for the current latest `devauc` and `sersic` runs.

    The wrapper keeps the operational workflow simple: once both inference runs
    finish, one CLI call can regenerate both required corner artifacts.
    """

    devauc_result = run_posterior_corner(run_dir=devauc_run_dir, burn_in=burn_in)
    sersic_result = run_posterior_corner(run_dir=sersic_run_dir, burn_in=burn_in)
    return PosteriorCornerLatestResult(
        status="completed",
        devauc_result=devauc_result,
        sersic_result=sersic_result,
        metadata={"burn_in_request": burn_in},
    )
