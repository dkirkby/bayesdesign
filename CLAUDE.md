# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `bayesdesign`, a Python package for Bayesian optimal experiment design. The package calculates expected information gain (EIG) for experimental design optimization. The core functionality is built around discrete grids for parameters, features, and experimental designs.

## Commands

### Testing
- Run all tests: `tox -e py`
- Run tests with pytest directly: `pytest tests`
- Run specific test file: `pytest tests/test_grid.py`

### Development Setup
- Install development dependencies: `pip install tox pytest black check-manifest`
- Install package in development mode: `pip install -e .`

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

The package is organized around three main modules in `src/bed/`:

1. **`grid.py`** - The foundation of the package providing:
   - `Grid` class: Discrete grids for variables with support for constraints
   - `GridStack` context manager: For broadcasting operations across multiple grids
   - Grid arithmetic and normalization operations
   - Constraint handling for complex parameter spaces

2. **`design.py`** - Experiment design optimization:
   - `ExperimentDesigner` class: Brute force EIG calculations
   - Memory management for large parameter spaces via subgrid chunking
   - Integration with Grid objects for parameters, features, and designs
   - Support for custom likelihood functions

3. **`plot.py`** - Visualization utilities for grids and results

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
4. Use `GridStack` context to define likelihood function relating all three
5. Create `ExperimentDesigner` with grids and likelihood
6. Calculate EIG for optimal experimental design selection

## Testing

Tests are located in `tests/` directory:
- `test_grid.py`: Tests for Grid class and GridStack functionality
- `test_design.py`: Tests for ExperimentDesigner class

The test suite uses pytest and covers core functionality including grid operations, constraint handling, and EIG calculations.

## Examples

Jupyter notebook examples are in the `examples/` directory:
- `SineWave.ipynb`: Basic sine wave parameter estimation
- `MultiParameter.ipynb`: Multiple parameter estimation
- `LocationFinding.ipynb`: Spatial optimization problems
- `GridConstraints.ipynb`: Working with constrained parameter spaces
- `Subgrids.ipynb`: Memory management with subgrids

## Package Structure

```
src/bed/
├── __init__.py          # Version info
├── grid.py             # Core Grid class and utilities
├── design.py           # ExperimentDesigner for EIG calculation
└── plot.py             # Plotting utilities
```

The package follows standard Python packaging conventions with setuptools, supports Python 3.6+, and has minimal dependencies (only numpy required, matplotlib optional for plotting).