# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `bayesdesign`, a Python package for Bayesian optimal experiment design. The package calculates expected information gain (EIG) for experimental design optimization. The core functionality is built around discrete grids for parameters, features, and experimental designs.

As of v0.7.0, the public grid and design APIs are JAX-backed. `Grid`, `GridStack`, prior helpers, and `ExperimentDesigner` use `jax.numpy` arrays with float64 enabled, support explicit CPU/GPU device placement, and include chunked design-space execution for larger runs. The package now requires Python 3.10+ and depends on both NumPy and JAX.

## Commands

### Testing
- Run all tests: `tox -e py`
- Run tests with pytest directly: `pytest tests`
- Run specific test file: `pytest tests/test_grid.py`

### Development Setup
- Install development dependencies: `pip install tox pytest black check-manifest`
- Install package in development mode: `pip install -e .`
- Install GPU-capable JAX extras when needed: `pip install -e ".[jax-cuda12]"` or `pip install -e ".[jax-cuda13]"`
- Install benchmark helpers: `pip install -e ".[benchmark]"` or `pip install -e ".[benchmark-gpu]"`

### Code Quality
- Format code with black: `black src/ tests/`
- Check package manifest: `check-manifest --ignore 'tox.ini,tests/**,.editorconfig,vscode.env,.vscode/**,examples/**'`
- Validate setup: `python setup.py check -m -s`

### Building and Distribution
- Build package: `python -m build`
- Install from source: `pip install .`

### Release Process
- Validate release readiness: `python scripts/validate_release.py`
- The release process uses GitHub Actions with trusted publishing (see `TRUSTED_PUBLISHING_SETUP.md`)
- Pre-release validation includes: tests, notebook execution, version consistency, changelog verification

## Architecture

### Core Components

The package is organized around these main modules in `src/bed/`:

1. **`grid.py`** - The foundation of the package providing:
   - `Grid` class: JAX-backed discrete grids for variables with support for constraints
   - `GridStack` context manager: For broadcasting operations across multiple grids
   - Grid arithmetic and normalization operations
   - Constraint handling for complex parameter spaces
   - Explicit device selection through `device="cpu"`, `device="gpu"`, or a `jax.Device`

2. **`design.py`** - Experiment design optimization:
   - `ExperimentDesigner` class: JAX/JIT-backed brute force EIG calculations
   - Memory management for large design spaces via `mem` or explicit `design_chunk_size` chunking
   - Integration with Grid objects for parameters, features, and designs
   - Support for custom unnormalized likelihood functions, passed as callables and normalized internally over features
   - Marginal EIG and posterior/update APIs that operate on JAX arrays

3. **`benchmark.py`** - Benchmark and profiling utilities:
   - Timing and process RSS sampling with the `benchmark` extra
   - Optional NVML/JAX device-memory sampling with the `benchmark-gpu` extra
   - Plotting helpers for benchmark memory time series

4. **`plot.py`** - Visualization utilities for grids and results

### Key Concepts

- **Parameters**: Variables being estimated (e.g., model parameters)
- **Features**: Observable quantities (e.g., measurements, data)
- **Designs**: Experimental configurations (e.g., observation times, conditions)
- **Likelihood**: Probability of observing features given parameters and design
- **EIG**: Expected Information Gain - the metric optimized for experimental design

### Typical Workflow

1. Define parameter space using `Grid(param1=values1, param2=values2, ...)`
2. Define feature space (observables) using `Grid(feature=values)`
3. Define design space (experimental configurations) using `Grid(design_var=values)`
4. Define a JAX-traceable unnormalized likelihood function:
   - Signature: `unnorm_lfunc(params, features, designs, **kwargs)`
   - Use `jax.numpy` (`jnp`) operations inside the function, especially when chunking is enabled
   - Pass scalar/config values through `lfunc_args={...}`
5. Create `ExperimentDesigner` with grids and the unnormalized likelihood function
6. Calculate EIG for optimal experimental design selection

Example skeleton:

```python
import jax.numpy as jnp

from bed.design import ExperimentDesigner
from bed.grid import Grid

designs = Grid(t_obs=jnp.linspace(0, 5, 51), device="cpu")
features = Grid(y_obs=jnp.linspace(-1.25, 1.25, 100), device="cpu")
params = Grid(
    amplitude=1.0,
    frequency=jnp.linspace(0.2, 2.0, 181),
    offset=0.0,
    device="cpu",
)

def unnorm_lfunc(params, features, designs, sigma_y):
    y_mean = params.amplitude * jnp.sin(
        params.frequency * (designs.t_obs - params.offset)
    )
    return jnp.exp(-0.5 * ((features.y_obs - y_mean) / sigma_y) ** 2)

designer = ExperimentDesigner(
    params,
    features,
    designs,
    unnorm_lfunc,
    lfunc_args={"sigma_y": 0.1},
    device="cpu",
)
prior = params.normalize(jnp.ones(params.shape))
best_design = designer.calculateEIG(prior)
```

Important JAX behavior:
- JAX arrays are immutable. `params.normalize(prior)` returns a normalized array; always assign the return value (`prior = params.normalize(prior)`).
- All grids passed to `ExperimentDesigner` must be on the same device as the designer.
- Chunked paths (`mem` or `design_chunk_size`) require the likelihood function to be JAX-traceable. Avoid regular `numpy` operations inside likelihood callables.
- The implementation enables `jax_enable_x64` in the core modules to preserve numerical precision.

## Testing

Tests are located in `tests/` directory:
- `test_grid.py`: Tests for Grid class and GridStack functionality
- `test_design.py`: Tests for ExperimentDesigner class
- `test_baseline.py`: Golden-value parity tests for the JAX implementation
- `test_runtime.py`: Device placement, runtime behavior, and JAX array tests
- `test_benchmark.py`: Benchmark helper tests

The test suite uses pytest and covers core functionality including grid operations, constraint handling, and EIG calculations.

## Examples

Jupyter notebook examples are in the `examples/` directory:
- `SineWave.ipynb`: Basic sine wave parameter estimation
- `MultiParameter.ipynb`: Multiple parameter estimation
- `LocationFinding.ipynb`: Spatial optimization problems
- `GridConstraints.ipynb`: Working with constrained parameter spaces
- `Subgrids.ipynb`: Memory management with subgrids
- `Benchmarking.ipynb`: Benchmarking workflow and historical NumPy-vs-JAX comparisons

Benchmark artifacts are stored under `docs/benchmarks/`, including historical sweeps and time-series memory traces used by the benchmarking notebook.

## Package Structure

```
src/bed/
├── __init__.py          # Version info
├── grid.py             # Core Grid class and utilities
├── design.py           # ExperimentDesigner for EIG calculation
├── benchmark.py        # Benchmark timing, memory tracing, and plotting helpers
├── util.py             # Shared JAX device helpers
└── plot.py             # Plotting utilities
```

The package follows standard Python packaging conventions with setuptools, supports Python 3.10+, and has minimal core dependencies (`numpy` and `jax`, with matplotlib optional for plotting).
