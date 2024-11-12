# Bayesian Optimal Experiment Design

[![PyPI package](https://img.shields.io/badge/pip%20install-bayesdesign-brightgreen)](https://pypi.org/project/bayesdesign/) [![GitHub Release](https://img.shields.io/github/v/release/dkirkby/bayesdesign?color=green)](https://github.com/dkirkby/bayesdesign/releases) [![Actions Status](https://github.com/dkirkby/bayesdesign/workflows/Test/badge.svg)](https://github.com/dkirkby/bayesdesign/actions) [![License](https://img.shields.io/github/license/dkirkby/bayesdesign)](https://github.com/dkirkby/bayesdesign/blob/main/LICENSE)

Use this package to calculate expected information gain for Bayesian optimal experiment design. For an introduction to this topic, see this [interactive notebook](https://observablehq.com/@dkirkby/boed). To perform a similar calculation with this package, use:
```python
from bed.grid import Grid, GridStack
from bed.design import ExperimentDesigner

designs = Grid(t_obs=np.linspace(0, 5, 51))
features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
params = Grid(amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0)

sigma_y=0.1
with GridStack(features, designs, params):
    y_mean = params.amplitude * np.sin(params.frequency * (designs.t_obs - params.offset))
    y_diff = features.y_obs - y_mean
    likelihood = np.exp(-0.5 * (y_diff / sigma_y) ** 2)
    features.normalize(likelihood)

designer = ExperimentDesigner(params, features, designs, likelihood)

prior = np.ones(params.shape)
params.normalize(prior);

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
The only required dependency is numpy. The optional plot module also requires matplotlib.

The changes with each version are documented [here](CHANGELOG.md).

## Upgrade

To upgrade your pip-installed package to the [latest released version](https://github.com/dkirkby/bayesdesign/releases/latest) use:
```
pip install bayesdesign --upgrade
```

## Contributing

If you have feedback or would like to contribute to this package, please see our [contributor's guide](CONTRIBUTING.md).
