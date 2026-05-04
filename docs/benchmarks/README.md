# Historical NumPy vs JAX Benchmarks

These benchmark artifacts preserve a NumPy-vs-JAX efficiency comparison before the 
migration of the bayesdesign code from NumPy to JAX. The goal is to document why 
moving `ExperimentDesigner.calculateEIG` to JAX helps as the parameter grid grows.

The comparison uses three recorded machines:

- `macbook`: local MacBook CPU used as development baseline
- `entropy`: server runs with CPU traces and JAX GPU device-memory traces
- `perlmutter`: HPC runs with CPU traces and JAX GPU device-memory traces

The basic finding is NumPy is often more efficient for tiny CPU jobs because it has 
little compilation/startup overhead. As the three-dimensional parameter grid gets 
larger, JAX becomes faster and often uses less peak CPU memory (RSS).

## CPU Scaling Trend

For small grids with the number of points per parameter axis set to just `n=10`, 
NumPy has the advantage. However, by `n=80`, JAX is consistently faster, 
and the peak RSS ratio also shifts toward JAX for most machine/configuration pairs.

Ratios below are `NumPy / JAX`, so values below `1.0x` favor NumPy and values
above `1.0x` favor JAX.

| Machine | Configuration | n=10 speed | n=80 speed | n=10 peak RSS mem | n=80 peak RSS mem |
|---|---:|---:|---:|---:|---:|
| macbook | full grid | 0.03x | 10.58x | 0.42x | 0.96x |
| macbook | subgrid | 0.06x | 2.81x | 0.04x | 1.31x |
| entropy | full grid | 0.02x | 2.66x | 0.24x | 2.12x |
| entropy | subgrid | 0.05x | 5.75x | 0.02x | 1.24x |
| perlmutter | full grid | 0.04x | 6.68x | 0.46x | 2.46x |
| perlmutter | subgrid | 0.11x | 3.52x | 0.03x | 1.08x |

## Time-Series Memory Traces

The memory traces live in `historical/timeseries/`. Each CSV represents one experiment 
configuration on one machine, for example:

- `historical/timeseries/macbook_full.csv`
- `historical/timeseries/macbook_subgrid.csv`
- `historical/timeseries/entropy_full.csv`
- `historical/timeseries/perlmutter_subgrid.csv`

The time axis is centered on the READY state:

- `time_bin_s < 0`: process startup / import / setup before the benchmarked run
- `time_bin_s = 0`: first sample at or after READY
- `time_bin_s > 0`: benchmarked run after READY

Memory columns are deltas from the READY memory state:

- `n80_rss_delta_mb_np`: NumPy process RSS delta for an 80×80×80 parameter grid
- `n80_rss_delta_mb_jax`: JAX process RSS delta for the same run
- `n80_gpu_delta_mb_jax`: JAX device-memory delta for GPU runs

This makes the plot answer: once the experiment is ready to execute, how much
additional memory does `calculateEIG` require?

The examples below use paths relative to this README in `docs/benchmarks/`.

```python
from pathlib import Path

from bed.benchmark import plot_timeseries

base = Path("historical/timeseries")
plots = Path("historical/plots")

labels = [
    "macbook NumPy",
    "macbook JAX",
    "entropy NumPy",
    "entropy JAX",
    "perlmutter NumPy",
    "perlmutter JAX",
]
value_cols = {
    "macbook NumPy": ["n80_rss_delta_mb_np"],
    "macbook JAX": ["n80_rss_delta_mb_jax"],
    "entropy NumPy": ["n80_rss_delta_mb_np"],
    "entropy JAX": ["n80_rss_delta_mb_jax"],
    "perlmutter NumPy": ["n80_rss_delta_mb_np"],
    "perlmutter JAX": ["n80_rss_delta_mb_jax"],
}
colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:green"]
linestyles = ["--", "-", "--", "-", "--", "-"]

plot_timeseries(
    [
        base / "macbook_full.csv",
        base / "macbook_full.csv",
        base / "entropy_full.csv",
        base / "entropy_full.csv",
        base / "perlmutter_full.csv",
        base / "perlmutter_full.csv",
    ],
    labels,
    plots / "timeseries_full_grid_memory_cpu.png",
    value_cols=value_cols,
    colors=colors,
    linestyles=linestyles,
    alpha=0.6,
    ylabel="RSS delta from READY (MiB)",
    title="Full-grid CPU memory time series, n=80",
)
```

Use the subgrid files to generate the equivalent chunked-design comparison:

```python
plot_timeseries(
    [
        base / "macbook_subgrid.csv",
        base / "macbook_subgrid.csv",
        base / "entropy_subgrid.csv",
        base / "entropy_subgrid.csv",
        base / "perlmutter_subgrid.csv",
        base / "perlmutter_subgrid.csv",
    ],
    labels,
    plots / "timeseries_subgrid_memory_cpu.png",
    value_cols=value_cols,
    colors=colors,
    linestyles=linestyles,
    alpha=0.6,
    ylabel="RSS delta from READY (MiB)",
    title="Subgrid CPU memory time series, n=80",
)
```

GPU traces use JAX device-memory measurements, not RSS:

```python
gpu_value_cols = {
    "entropy GPU": ["n80_gpu_delta_mb_jax"],
    "perlmutter GPU": ["n80_gpu_delta_mb_jax"],
}
gpu_labels = ["entropy GPU", "perlmutter GPU"]
gpu_colors = ["tab:orange", "tab:green"]

plot_timeseries(
    [base / "entropy_full.csv", base / "perlmutter_full.csv"],
    gpu_labels,
    plots / "timeseries_full_grid_memory_gpu.png",
    value_cols=gpu_value_cols,
    colors=gpu_colors,
    alpha=0.6,
    ylabel="JAX GPU memory delta from READY (MiB)",
    title="Full-grid JAX GPU memory time series, n=80",
)
```

## Peak Memory Sweeps

The `historical/sweeps/` tables summarize peak memory deltas over
parameter-grid size. They use the same READY-relative convention as the
time-series data.

Example files:

- `historical/sweeps/macbook_full_peak.csv`
- `historical/sweeps/macbook_subgrid_peak.csv`
- `historical/sweeps/entropy_full_peak.csv`
- `historical/sweeps/perlmutter_subgrid_peak.csv`

Columns:

- `n_param_axis`: number of points in each of the three parameter axes
- `rss_delta_mb_np`: NumPy peak RSS delta from READY
- `rss_delta_mb_jax`: JAX peak RSS delta from READY
- `gpu_delta_mb_jax`: JAX GPU/device peak delta from READY, when available

These CSVs are intentionally plain tables so project-specific comparison plots
can be regenerated with regular `pandas`/`matplotlib` code. The installable
benchmark API stays focused on profiling individual runs and combining their
memory traces; peak-memory comparisons are easier to configure explicitly for a
given experiment.

## Regenerating the Curated CSVs

The curated CSVs are generated from the old raw `benchmark_results/` traces:

```bash
conda run -n bayesdesign python prepare_existing_benchmark_data.py \
    --source ../../benchmark_results \
    --output historical
```

The converter intentionally keeps only:

- named machines: `macbook`, `entropy`, and `perlmutter`
- 3D sine-wave benchmarks
- full-grid and subgrid chunk-1 configurations

This avoids preserving older exploratory tables that duplicated the same data
with less useful schemas.
