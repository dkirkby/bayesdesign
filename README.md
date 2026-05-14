# Bayesian Optimal Experiment Design

[![PyPI package](https://img.shields.io/badge/pip%20install-bayesdesign-brightgreen)](https://pypi.org/project/bayesdesign/) [![GitHub Release](https://img.shields.io/github/v/release/dkirkby/bayesdesign?color=green)](https://github.com/dkirkby/bayesdesign/releases) [![Actions Status](https://github.com/dkirkby/bayesdesign/workflows/Test/badge.svg)](https://github.com/dkirkby/bayesdesign/actions) [![License](https://img.shields.io/github/license/dkirkby/bayesdesign)](https://github.com/dkirkby/bayesdesign/blob/main/LICENSE)

Use this package to calculate expected information gain for Bayesian optimal experiment design. For an introduction to this topic, see this [interactive notebook](https://observablehq.com/@dkirkby/boed). To perform a similar calculation with this package, use:
```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

from bed.grid import Grid
from bed.design import ExperimentDesigner

designs = Grid(t_obs=jnp.linspace(0, 5, 51))
features = Grid(y_obs=jnp.linspace(-1.25, 1.25, 100))
params = Grid(amplitude=1, frequency=jnp.linspace(0.2, 2.0, 181), offset=0)

def unnorm_lfunc(params, features, designs, sigma_y):
    y_mean = params.amplitude * jnp.sin(
        params.frequency * (designs.t_obs - params.offset)
    )
    y_diff = features.y_obs - y_mean
    return jnp.exp(-0.5 * (y_diff / sigma_y) ** 2)

designer = ExperimentDesigner(
    params,
    features,
    designs,
    unnorm_lfunc,
    lfunc_args={"sigma_y": 0.1},
)

prior = params.normalize(jnp.ones(params.shape))

designer.calculateEIG(prior)

plt.plot(designs.t_obs, designer.EIG)
```

Browse the [examples folder](https://github.com/dkirkby/bayesdesign/) to learn more about using this package.

To run the examples in [google colab](https://colab.research.google.com/), select **GitHub** and enter `https://github.com/dkirkby/bayesdesign`.

## Installation

Install the [latest released version](https://github.com/dkirkby/bayesdesign/releases/latest) from [pypi](https://pypi.org/project/bayesdesign/) using:
```
pip install bayesdesign
```
Python 3.10 or newer is required. The core dependencies are NumPy and JAX. Install the optional CUDA-enabled JAX extras when you want GPU execution:
```
pip install "bayesdesign[jax-cuda12]"
# or
pip install "bayesdesign[jax-cuda13]"
```

The optional plot module also requires matplotlib. Benchmarking helpers are available with:
```
pip install "bayesdesign[benchmark]"
pip install "bayesdesign[benchmark-gpu]"
```

## JAX execution

Version 0.7.0 migrated the public grid and design APIs to JAX while preserving the package's discrete-grid workflow. `Grid`, `GridStack`, priors, and `ExperimentDesigner` now use JAX arrays with float64 enabled.

Device placement can be selected explicitly:
```python
params = Grid(device="cpu", frequency=jnp.linspace(0.2, 2.0, 181))
# or, with a GPU-enabled JAX install:
params = Grid(device="gpu", frequency=jnp.linspace(0.2, 2.0, 181))
```

All grids passed to an `ExperimentDesigner` must live on the same device, or the designer must be constructed with a matching `device=`. Large design spaces can be evaluated in chunks with either the legacy memory limit or an explicit design chunk size:
```python
designer = ExperimentDesigner(
    params,
    features,
    designs,
    unnorm_lfunc,
    lfunc_args={"sigma_y": 0.1},
    design_chunk_size=128,
)
```

When chunking is used, the likelihood function must be JAX-traceable: use `jax.numpy` operations inside `unnorm_lfunc`, not regular `numpy` operations.

The changes with each version are documented [here](CHANGELOG.md).

## Upgrade

To upgrade your pip-installed package to the [latest released version](https://github.com/dkirkby/bayesdesign/releases/latest) use:
```
pip install bayesdesign --upgrade
```

## Contributing

If you have feedback or would like to contribute to this package, please see our [contributor's guide](CONTRIBUTING.md).
