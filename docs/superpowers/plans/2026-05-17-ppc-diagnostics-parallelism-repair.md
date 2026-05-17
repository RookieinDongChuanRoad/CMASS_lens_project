# PPC Diagnostics Adapter Numba Parallelism Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore real PPC diagnostics parallelism by making every model adapter own a Numba-optimized diagnostics hot path, while keeping generic PPC workflow code model-agnostic.

**Architecture:** `predictive.py` remains responsible for loading a completed run, selecting posterior draws, resolving a diagnostics execution policy, and writing artifacts. Each `adapters/*.py` module remains responsible for model-specific posterior-predictive semantics and must expose a Numba-backed diagnostics implementation, following the current CMASS pattern. The existing `--worker-processes` CLI/API knob must no longer be discarded; for the first repair pass it is treated as requested diagnostics compute width and mapped to Numba kernel threads under the default `kernel_only` policy, with process-pool chunking left as a measured second pass rather than the primary fix.

**Tech Stack:** Python 3, `cmass_lens` conda environment, `pytest`, Numba, NumPy, existing `cmass_lens_inference.parallel.apply_thread_limits`, current PPC `PredictiveDefinition` registry.

---

## File Structure

- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/interfaces.py`
  - Add a diagnostics execution contract shared by all model adapters.
- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
  - Stop discarding `worker_processes`.
  - Resolve diagnostics parallelism once.
  - Pass execution context to `predictive_definition.run_diagnostics`.
  - Write truthful parallel metadata into PPC artifacts.
- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
  - Accept execution context.
  - Keep the existing Numba shared-parent kernel as the canonical adapter pattern.
  - Apply resolved Numba thread limits before entering the diagnostics kernel.
- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/sonnenfeld.py`
  - Replace the current Python draw/sample loops with a CMASS-style Numba diagnostics kernel.
  - Keep the paper-native and sigma-star-gamma model branches inside the adapter, not in generic PPC code.
  - Use chunked random-array generation outside the kernel, then run the heavy population loop inside Numba.
- Modify: `Posterior_predictive_test/tests/test_predictive_registry.py`
  - Lock the adapter contract so future models must accept diagnostics execution context and advertise a Numba backend.
- Modify: `Posterior_predictive_test/tests/test_posterior_predictive.py`
  - Replace stale `worker_processes == 0` assertions with execution-policy assertions.
  - Add regression tests for CMASS and Sonnenfeld Numba-backed PPC diagnostics.
- Optional docs update: `Posterior_predictive_test/README.md`
  - Document that PPC parallelism is adapter-owned Numba parallelism by default; process pools are optional and must be benchmarked per adapter.

## Design Decisions

- Do not move scientific prediction logic out of adapters. Adapters own how a posterior draw becomes replicated lens-population diagnostics.
- Do not implement Sonnenfeld by wrapping the current Python loop in a process pool as the main fix. That would parallelize a slow implementation instead of fixing the model adapter to match the CMASS Numba pattern.
- Do interpret the existing `--worker-processes` knob as a user request for diagnostics compute width. Under default `kernel_only`, the artifact should record `requested_worker_processes=<requested>`, `worker_processes=0`, and `kernel_threads_per_process=<resolved>`.
- Do allow a future `process_pool` mode, but only after all adapters have Numba kernels. In that mode each process must call the same adapter-owned Numba chunk runner with `kernel_threads_per_process=1` to avoid CPU oversubscription.
- Do not introduce CMASS or Sonnenfeld branches into `predictive.py`. The generic layer passes a resolved execution object; the adapter decides how to use it.
- Do not run full-chain Sonnenfeld PPC as the first validation. Use unit tests, tiny real-run smoke, then a medium benchmark before any full-chain rerun.

---

### Task 1: Add Diagnostics Execution Contract

**Files:**
- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/interfaces.py`
- Test: `Posterior_predictive_test/tests/test_predictive_registry.py`

- [ ] **Step 1: Write the failing contract test**

Add this test to `Posterior_predictive_test/tests/test_predictive_registry.py`:

```python
def test_predictive_definition_diagnostics_hook_accepts_execution_context() -> None:
    """Every model adapter should receive resolved PPC diagnostics execution metadata."""

    import inspect

    from lensing_posterior_predictive.registry import get_predictive_definition

    for model_name in (
        "cmass",
        "sonnenfeld2024_slacs",
        "sonnenfeld2024_slacs_hunit",
        "sonnenfeld2024_slacs_sigma_star_gamma",
        "sonnenfeld2024_slacs_sigma_star_gamma_hunit",
    ):
        definition = get_predictive_definition(model_name)
        signature = inspect.signature(definition.run_diagnostics)
        assert "execution" in signature.parameters
        assert definition.backend.startswith("numba")
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py::test_predictive_definition_diagnostics_hook_accepts_execution_context -q
```

Expected: FAIL because current adapter hooks do not expose `execution`.

- [ ] **Step 3: Add execution dataclass**

In `Posterior_predictive_test/src/lensing_posterior_predictive/interfaces.py`, add:

```python
@dataclass(frozen=True)
class DiagnosticsExecution:
    """
    Resolved execution policy for one PPC diagnostics run.

    The generic PPC layer resolves CPU budget and records artifact metadata.
    Model adapters own how that budget is consumed.  The default policy is
    Numba kernel parallelism inside the adapter, not generic Python process
    parallelism around model-specific code.
    """

    strategy: str
    cpu_count: int
    reserve_cores: int
    compute_budget: int
    requested_worker_processes: int | None
    worker_processes: int
    kernel_threads_per_process: int

    def to_dict(self) -> dict[str, int | str | None]:
        """Serialize diagnostics execution metadata into PPC artifacts."""

        return {
            "strategy": self.strategy,
            "cpu_count": self.cpu_count,
            "reserve_cores": self.reserve_cores,
            "compute_budget": self.compute_budget,
            "requested_worker_processes": self.requested_worker_processes,
            "worker_processes": self.worker_processes,
            "kernel_threads_per_process": self.kernel_threads_per_process,
        }
```

Update `__all__`:

```python
__all__ = ["DiagnosticsExecution", "PPCContextBundle", "PredictiveDefinition"]
```

- [ ] **Step 4: Update adapter hook signatures**

In `adapters/cmass.py`, change `_run_shared_parent_diagnostics_numba(...)` to accept:

```python
    *,
    execution: DiagnosticsExecution,
) -> dict[str, Any]:
```

In `adapters/sonnenfeld.py`, change the public `_run_sonnenfeld_parent_diagnostics(...)` wrapper to accept the same keyword-only argument.

- [ ] **Step 5: Run the contract test**

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py::test_predictive_definition_diagnostics_hook_accepts_execution_context -q
```

Expected: PASS.

---

### Task 2: Resolve Diagnostics Compute Width In Generic PPC

**Files:**
- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/predictive.py`
- Test: `Posterior_predictive_test/tests/test_posterior_predictive.py`

- [ ] **Step 1: Add failing metadata test**

Add this test near the existing posterior-diagnostics tests:

```python
def test_run_posterior_diagnostics_records_requested_compute_width(tmp_path: Path) -> None:
    """PPC diagnostics should not silently discard the user-requested compute width."""

    from lensing_posterior_predictive.predictive import run_posterior_diagnostics

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="devauc")

    result = run_posterior_diagnostics(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(tmp_path / "diagnostics_output"),
        n_posterior_draws=3,
        burn_in=1,
        random_seed=131,
        parent_sample_size=72,
        n_mass_bins=5,
        mass_bin_min=10.9,
        mass_bin_max=11.9,
        worker_processes=2,
    )

    ppc_summary = json.loads((result.result_dir / "ppc_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((result.result_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert ppc_summary["parallelism"]["strategy"] == "kernel_only"
    assert ppc_summary["parallelism"]["requested_worker_processes"] == 2
    assert ppc_summary["parallelism"]["worker_processes"] == 0
    assert ppc_summary["parallelism"]["kernel_threads_per_process"] == 2
    assert manifest["parallelism"] == ppc_summary["parallelism"]
    assert result.metadata["parallelism"] == ppc_summary["parallelism"]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_posterior_predictive.py::test_run_posterior_diagnostics_records_requested_compute_width -q
```

Expected: FAIL because current code deletes `worker_processes` and hard-codes parallel metadata.

- [ ] **Step 3: Implement resolver**

In `predictive.py`, import `DiagnosticsExecution`:

```python
from .interfaces import DiagnosticsExecution
```

Add helper near existing PPC runtime helpers:

```python
def _resolve_diagnostics_execution(
    runtime_config: RuntimeConfig,
    requested_worker_processes: int | None,
    n_draws: int,
) -> DiagnosticsExecution:
    """
    Resolve the PPC diagnostics compute policy.

    The current production policy is adapter-owned Numba kernel parallelism.
    The historical `worker_processes` API is therefore treated as requested
    compute width unless a future explicit process-pool mode is added and
    benchmarked per adapter.
    """

    cpu_count = max(1, int(os.cpu_count() or 1))
    reserve_cores = max(0, int(runtime_config.runtime.reserve_cores))
    auto_budget = max(1, cpu_count - reserve_cores)
    configured_threads = int(runtime_config.runtime.num_threads)
    compute_budget = auto_budget if configured_threads <= 0 else max(1, min(configured_threads, auto_budget))

    if requested_worker_processes is None:
        requested_width = compute_budget
        serialized_request = None
    else:
        requested_width = max(1, int(requested_worker_processes))
        serialized_request = requested_width

    kernel_threads = max(1, min(int(requested_width), compute_budget, max(1, int(n_draws))))
    return DiagnosticsExecution(
        strategy="kernel_only",
        cpu_count=cpu_count,
        reserve_cores=reserve_cores,
        compute_budget=compute_budget,
        requested_worker_processes=serialized_request,
        worker_processes=0,
        kernel_threads_per_process=kernel_threads,
    )
```

- [ ] **Step 4: Wire resolver into `run_posterior_diagnostics`**

Remove:

```python
    del worker_processes
```

After `n_posterior_draws_used = int(posterior_draws.shape[0])`, add:

```python
    execution = _resolve_diagnostics_execution(
        runtime_config=runtime_config,
        requested_worker_processes=worker_processes,
        n_draws=n_posterior_draws_used,
    )
```

Pass it to the adapter call:

```python
        random_seed=int(random_seed),
        execution=execution,
```

Replace the hard-coded `parallelism_payload` with:

```python
    parallelism_payload = execution.to_dict()
```

- [ ] **Step 5: Run the metadata test**

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_posterior_predictive.py::test_run_posterior_diagnostics_records_requested_compute_width -q
```

Expected: PASS.

---

### Task 3: Make CMASS The Explicit Adapter Pattern

**Files:**
- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/cmass.py`
- Test: `Posterior_predictive_test/tests/test_posterior_predictive.py`

- [ ] **Step 1: Add CMASS regression test**

Add:

```python
def test_cmass_diagnostics_uses_numba_kernel_threads(tmp_path: Path) -> None:
    """CMASS should keep its adapter-owned Numba kernel and consume resolved thread width."""

    from lensing_posterior_predictive.predictive import run_posterior_diagnostics

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="devauc")

    result = run_posterior_diagnostics(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(tmp_path / "diagnostics_output"),
        n_posterior_draws=3,
        burn_in=1,
        random_seed=41,
        parent_sample_size=72,
        n_mass_bins=5,
        mass_bin_min=10.9,
        mass_bin_max=11.9,
        worker_processes=2,
    )

    ppc_summary = json.loads((result.result_dir / "ppc_summary.json").read_text(encoding="utf-8"))

    assert ppc_summary["backend"] == "numba_shared_parent"
    assert ppc_summary["parallelism"]["strategy"] == "kernel_only"
    assert ppc_summary["parallelism"]["kernel_threads_per_process"] == 2
    assert (result.result_dir / "replicated_statistics.npz").exists()
```

- [ ] **Step 2: Apply thread limit in CMASS adapter**

At the start of `_run_shared_parent_diagnostics_numba`, before random chunk generation, add:

```python
    apply_thread_limits(int(execution.kernel_threads_per_process))
```

Import:

```python
from cmass_lens_inference.parallel import apply_thread_limits
from ..interfaces import DiagnosticsExecution, PPCContextBundle, PredictiveDefinition
```

Add a short comment explaining why this is adapter-local:

```python
    # The CMASS adapter already owns a `parallel=True` Numba kernel.  The
    # generic PPC layer resolves the desired compute width; the adapter applies
    # it to Numba before entering the model-specific hot path.
```

- [ ] **Step 3: Run focused CMASS diagnostics tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest \
  Posterior_predictive_test/tests/test_posterior_predictive.py::test_run_posterior_diagnostics_records_requested_compute_width \
  Posterior_predictive_test/tests/test_posterior_predictive.py::test_cmass_diagnostics_uses_numba_kernel_threads \
  -q
```

Expected: PASS.

---

### Task 4: Replace Sonnenfeld Python Loops With Numba Diagnostics Kernel

**Files:**
- Modify: `Posterior_predictive_test/src/lensing_posterior_predictive/adapters/sonnenfeld.py`
- Test: `Posterior_predictive_test/tests/test_predictive_registry.py`
- Test: `Posterior_predictive_test/tests/test_posterior_predictive.py`

- [ ] **Step 1: Add Sonnenfeld adapter contract test**

Add a test that directly asserts the Sonnenfeld definition advertises a Numba backend:

```python
def test_sonnenfeld_predictive_definitions_use_numba_backend() -> None:
    """Sonnenfeld PPC should not regress to Python-loop diagnostics."""

    from lensing_posterior_predictive.registry import get_predictive_definition

    for model_name in (
        "sonnenfeld2024_slacs",
        "sonnenfeld2024_slacs_hunit",
        "sonnenfeld2024_slacs_sigma_star_gamma",
        "sonnenfeld2024_slacs_sigma_star_gamma_hunit",
    ):
        definition = get_predictive_definition(model_name)
        assert definition.backend == "numba_sonnenfeld_parent"
```

- [ ] **Step 2: Create Numba output schema**

In `adapters/sonnenfeld.py`, define a `_SONNENFELD_DIAGNOSTIC_ARRAY_NAMES` tuple parallel to CMASS. It should include:

```python
(
    "theta_theta_ein",
    "theta_gamma",
    "theta_zd",
    "theta_zs",
    "theta_mass",
    "theta_re_kpc",
    "theta_n",
    "sigma_sigma",
    "sigma_theta_ein",
    "sigma_gamma",
    "sigma_zd",
    "sigma_zs",
    "sigma_mass",
    "sigma_re_kpc",
    "sigma_n",
    "theta_stats",
    "sigma_stats",
    "mass_trend",
    "gamma_trend",
    "sigma_trend",
    "parent_bin_counts",
    "detectable_weight_sums",
    "selected_weight_sums",
    "gamma_logre_trend",
    "gamma_logre_parent_bin_counts",
    "gamma_logre_detectable_weight_sums",
    "gamma_logre_selected_weight_sums",
    "gamma_sigma_star_trend",
    "gamma_sigma_star_parent_bin_counts",
    "gamma_sigma_star_detectable_weight_sums",
    "gamma_sigma_star_selected_weight_sums",
    "gamma_delta_r_trend",
    "gamma_delta_r_parent_bin_counts",
    "gamma_delta_r_detectable_weight_sums",
    "gamma_delta_r_selected_weight_sums",
)
```

- [ ] **Step 3: Add Numba helper functions**

Add local `@nb.njit(cache=True)` helpers for:

- percentile from sorted values
- summary statistics
- weighted-index draw from non-negative weights
- binned parent/detectable/selected means

Use the CMASS helper implementations as the behavioral template. Do not import private CMASS helper functions in this first pass; keep CMASS stable while Sonnenfeld is being ported.

- [ ] **Step 4: Pack Sonnenfeld context into Numba arrays**

Add a Python helper:

```python
def _sonnenfeld_context_numba_arrays(context) -> dict[str, np.ndarray | float | int]:
    """
    Extract the compiled Sonnenfeld context into Numba-compatible arrays.

    Numba kernels should receive plain arrays and scalar values, not the Python
    compiled-context object.  This keeps the hot path explicit and makes later
    process-pool chunking possible if benchmarks justify it.
    """
```

The helper must include every array/scalar currently read inside the Python loop:

- parent arrays: `parent_sample_zd`, `parent_sample_mstar`, `parent_sample_log_re`, `parent_sample_delta_r`, `base_normals`
- scalar model context: pivots, truncation bounds, `n_fixed`, `use_sersic_index`, source-redshift gap, mass-radius settings, sigma-proxy scatter
- cosmology/lensing grids: `z_grid`, `chi_kpc_grid`
- population sigma grid axes and values
- cross-section axes and grid

- [ ] **Step 5: Generate random chunk inputs outside the kernel**

Mirror CMASS chunking rather than allocating all full-chain random arrays at once:

```python
DEFAULT_SONNENFELD_NUMBA_DIAGNOSTICS_CHUNK_SIZE = 16
```

For each chunk:

- generate `parent_indices` row by row with the same replacement rule as current code
- generate `theta_uniforms` with shape `(chunk_size, THETA_SAMPLE_SIZE)`
- generate `sigma_uniforms` with shape `(chunk_size, SIGMA_SAMPLE_SIZE)`

Keep this wrapper in Python. The expensive per-parent physics and bin reductions belong in Numba.

- [ ] **Step 6: Implement `_sonnenfeld_parent_diagnostics_numba_chunk`**

Add:

```python
@nb.njit(cache=True, parallel=True)
def _sonnenfeld_parent_diagnostics_numba_chunk(...):
    """Evaluate Sonnenfeld PPC diagnostics for one posterior-draw chunk."""
```

The kernel should:

- loop over posterior draws with `nb.prange`
- preserve the 12D paper-gamma branch and 11D sigma-star-gamma branch
- compute `log_mass`, `gamma`, `z_s`, `theta_e`, `sigma_model`, and selection weights using the same helper kernels as the current Python implementation
- draw `theta_latent` and `sigma_latent` samples using pre-generated uniforms
- compute theta/sigma summary statistics
- compute mass-binned trends and the gamma-vs-logRe, gamma-vs-sigma-star, gamma-vs-delta-r trends
- return the same array tuple shape used by the current artifact writer

- [ ] **Step 7: Adapt Numba arrays back to PPC payload**

Replace the current `_run_sonnenfeld_parent_diagnostics(...)` body with a wrapper that:

1. calls `apply_thread_limits(int(execution.kernel_threads_per_process))`
2. chunks posterior draws
3. calls `_sonnenfeld_parent_diagnostics_numba_chunk(...)`
4. concatenates chunk outputs
5. maps arrays back into the existing payload keys:
   - `theta_latent`
   - `sigma_latent`
   - `theta_replicated_stats`
   - `sigma_replicated_stats`
   - `trend_draws`
   - all gamma-trend draw/count/weight arrays

The public payload schema must stay compatible with existing artifact writers.

- [ ] **Step 8: Add small Sonnenfeld smoke test**

Use an existing Sonnenfeld run fixture if available. If no fixture exists, add the smallest direct-adapter test that can build a real compiled Sonnenfeld context from the repo's test config and run:

```python
result = definition.run_diagnostics(
    posterior_draws=posterior_draws[:2],
    profile=profile,
    context=context,
    mass_definition=mass_definition,
    sigma_table=None,
    mass_bin_edges=mass_bin_edges,
    sigma_star_bin_edges=sigma_star_bin_edges,
    log_re_bin_edges=log_re_bin_edges,
    delta_r_bin_edges=delta_r_bin_edges,
    parent_sample_size=16,
    random_seed=123,
    execution=DiagnosticsExecution(
        strategy="kernel_only",
        cpu_count=2,
        reserve_cores=0,
        compute_budget=2,
        requested_worker_processes=2,
        worker_processes=0,
        kernel_threads_per_process=2,
    ),
)
```

Assert schema and shapes, not byte-identical stochastic values:

```python
assert result["theta_latent"]["theta_ein"].shape == (2, THETA_SAMPLE_SIZE)
assert result["sigma_latent"]["sigma"].shape == (2, SIGMA_SAMPLE_SIZE)
assert result["trend_draws"]["gamma"]["parent"].shape[0] == 2
```

- [ ] **Step 9: Run focused Sonnenfeld tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py -q
```

Then run the focused Sonnenfeld smoke test added above.

Expected: PASS.

---

### Task 5: Correct Stale Metadata Expectations

**Files:**
- Modify: `Posterior_predictive_test/tests/test_posterior_predictive.py`

- [ ] **Step 1: Replace stale `worker_processes == 0` assertions**

Search:

```bash
rg -n 'worker_processes.*== 0|\\["worker_processes"\\] == 0' Posterior_predictive_test/tests/test_posterior_predictive.py
```

For default kernel-only diagnostics, assert the full policy instead of a bare zero:

```python
assert payload["parallelism"]["strategy"] == "kernel_only"
assert payload["parallelism"]["worker_processes"] == 0
assert payload["parallelism"]["kernel_threads_per_process"] >= 1
```

For tests that pass `worker_processes=2`, assert:

```python
assert payload["parallelism"]["requested_worker_processes"] == 2
assert payload["parallelism"]["strategy"] == "kernel_only"
assert payload["parallelism"]["worker_processes"] == 0
assert payload["parallelism"]["kernel_threads_per_process"] == 2
```

- [ ] **Step 2: Remove misleading process-pool expectations**

Do not add tests that expect `worker_processes == 2` under the default policy. That would reintroduce the old confusion between process parallelism and Numba kernel parallelism.

- [ ] **Step 3: Run all PPC tests**

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py Posterior_predictive_test/tests/test_posterior_predictive.py -q
```

Expected: PASS.

---

### Task 6: Real-Run Smoke Validation

**Files:**
- No production source changes beyond Tasks 1-5.
- Writes artifacts under a temporary output root.

- [ ] **Step 1: Run CMASS smoke with explicit compute width**

Use a small completed CMASS run if available. Command shape:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m lensing_posterior_predictive.cli posterior-diagnostics \
  --run-dir /path/to/completed/cmass/run \
  --sigma-table /Users/liurongfu/Work/CMASS_lens_project/data/external/hunits_v1/jeans_deV_sigma_bundle.h5 \
  --output-dir /tmp/ppc_numba_parallel_smoke \
  --n-posterior-draws 4 \
  --parent-sample-size 64 \
  --worker-processes 2 \
  --burn-in auto
```

Expected metadata:

```json
{
  "parallelism": {
    "strategy": "kernel_only",
    "requested_worker_processes": 2,
    "worker_processes": 0,
    "kernel_threads_per_process": 2
  }
}
```

- [ ] **Step 2: Run Sonnenfeld smoke with explicit compute width**

Use the current h-unit sigma-star-gamma run with tiny posterior draw count:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m lensing_posterior_predictive.cli posterior-diagnostics \
  --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/20260511_130710_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit \
  --output-dir /tmp/ppc_numba_parallel_smoke \
  --n-posterior-draws 4 \
  --parent-sample-size 64 \
  --worker-processes 2 \
  --burn-in auto
```

Expected: command completes, `backend == "numba_sonnenfeld_parent"`, and metadata matches the kernel-only policy above.

- [ ] **Step 3: Verify smoke artifacts**

Run:

```bash
conda run -n cmass_lens --no-capture-output python - <<'PY'
from pathlib import Path
import json

root = Path("/tmp/ppc_numba_parallel_smoke")
for summary in root.glob("**/ppc_summary.json"):
    payload = json.loads(summary.read_text())
    print(summary)
    print(payload["backend"], payload["n_posterior_draws_used"], payload["parallelism"])
PY
```

Expected: each smoke summary reports `n_posterior_draws_used == 4`, a `numba_*` backend, and `kernel_threads_per_process == 2`.

---

### Task 7: Medium Benchmark Before Full-Chain PPC

**Files:**
- No production source changes.
- Optional output: `outputs/diagnostics_benchmarks/ppc_numba_parallel_YYYYMMDD.json`

- [ ] **Step 1: Benchmark Sonnenfeld with one kernel thread**

Run the h-unit sigma-star-gamma PPC smoke at a medium size:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m lensing_posterior_predictive.cli posterior-diagnostics \
  --run-dir /Users/liurongfu/Work/CMASS_lens_project/outputs/devauc/20260511_130710_devauc_sonnenfeld2024-slacs-sigma-star-gamma-hunit \
  --output-dir /tmp/ppc_numba_parallel_benchmark/thread1 \
  --n-posterior-draws 64 \
  --parent-sample-size 512 \
  --worker-processes 1 \
  --burn-in auto
```

- [ ] **Step 2: Benchmark Sonnenfeld with four kernel threads**

Run the same command with:

```bash
  --output-dir /tmp/ppc_numba_parallel_benchmark/thread4 \
  --worker-processes 4
```

- [ ] **Step 3: Compare wall time and artifacts**

Record:

- wall time
- `parallelism` payload
- whether artifacts are schema-compatible
- whether the speedup justifies full-chain PPC

Do not require an exact fixed speedup in CI; local CPU load makes that too brittle. The benchmark is a go/no-go check before launching a 192000-draw full-chain PPC.

---

### Task 8: Documentation Update

**Files:**
- Modify: `Posterior_predictive_test/README.md`

- [ ] **Step 1: Add execution policy section**

Append under Architecture:

```markdown
## Execution Policy

`posterior-diagnostics` resolves a diagnostics execution policy from
`--worker-processes`, CPU count, and the run's `runtime.reserve_cores`.
The default strategy is `kernel_only`: model adapters run their own Numba
diagnostics kernels and the requested worker count is used as Numba thread
width, not as a Python process count.

The generic workflow owns run loading, draw selection, and artifact metadata.
Model adapters own numerical prediction logic and optimization.  A process-pool
strategy may be added later per adapter, but only around an already Numba-backed
chunk runner and only after benchmark evidence shows it helps.
```

- [ ] **Step 2: Run README grep sanity**

Run:

```bash
rg -n "Execution Policy|kernel_only|worker_processes|Numba|process-pool" Posterior_predictive_test/README.md
```

Expected: the new section is visible.

---

## Final Verification

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_predictive_registry.py -q
```

Run:

```bash
conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PWD/Bayesian_inference/src:$PWD/Posterior_predictive_test/src:$PWD/prepare_dataset:/Users/liurongfu/tools" \
  python -m pytest Posterior_predictive_test/tests/test_posterior_predictive.py -q
```

Run the two real-run smoke commands from Task 6.

Run the medium Sonnenfeld benchmark from Task 7 before any full-chain PPC.

---

## Risk Notes

- Numba cannot consume the Python compiled-context object directly. Sonnenfeld must explicitly pack arrays and scalars before calling the kernel.
- The current Sonnenfeld implementation samples parent indices from canonical parent arrays. To avoid huge memory allocations, generate parent indices per chunk, not for the full 192000-draw chain at once.
- Python-side random chunk generation can still cost time, but it is not the main physics loop. If it becomes measurable after the Numba port, optimize that separately.
- Mapping `--worker-processes` to Numba thread width is intentionally conservative. It fixes the current "argument is discarded" bug without introducing process-pool pickling and oversubscription risks.
- A future process-pool strategy should not wrap Python loops. It should split posterior-draw chunks across processes, clamp each process to one Numba thread, and call the same adapter Numba chunk runner.
- Current full-chain Sonnenfeld PPC is too large for casual validation. Small and medium runs must pass before launching it.
