"""Shared pytest fixtures for baseline compatibility tests."""

import numpy as np
import pytest

from bed.grid import Grid, GridStack, TopHat
from bed.design import ExperimentDesigner


def _sine_wave_lfunc(params, features, designs, **kwargs):
    """Gaussian likelihood for sine wave model."""
    y_mean = params.amplitude * np.sin(
        params.frequency * (designs.t_obs - params.offset)
    )
    y_diff = features.y_obs - y_mean
    return np.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)


@pytest.fixture(scope="module")
def sine_wave_designer():
    """Sine wave scenario: 1D frequency estimation with fixed amplitude and offset."""
    designs = Grid(t_obs=np.linspace(0, 5, 51))
    features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
    params = Grid(
        amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0
    )

    designer = ExperimentDesigner(
        params, features, designs, _sine_wave_lfunc,
        lfunc_args={"sigma_y": 0.1},
    )

    prior = np.ones(params.shape)
    params.normalize(prior)
    designer.calculateEIG(prior)

    return {
        "designer": designer,
        "prior": prior,
        "params": params,
        "features": features,
        "designs": designs,
    }


@pytest.fixture(scope="module")
def sine_wave_designer_subgrid():
    """Sine wave scenario with mem=3 (subgrid chunking)."""
    designs = Grid(t_obs=np.linspace(0, 5, 51))
    features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
    params = Grid(
        amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0
    )

    designer = ExperimentDesigner(
        params, features, designs, _sine_wave_lfunc,
        lfunc_args={"sigma_y": 0.1},
        mem=3,
    )

    prior = np.ones(params.shape)
    params.normalize(prior)
    designer.calculateEIG(prior)

    return {
        "designer": designer,
        "prior": prior,
        "params": params,
        "features": features,
        "designs": designs,
    }


@pytest.fixture(scope="module")
def multi_param_designer():
    """Multi-parameter scenario: 3D parameter grid (amplitude, frequency, offset)."""
    designs = Grid(t_obs=np.linspace(0, 4, 32))
    features = Grid(y_obs=np.linspace(-1.4, 1.4, 40))
    params = Grid(
        amplitude=np.linspace(0.5, 1.5, 11),
        frequency=np.linspace(0.2, 2.0, 11),
        offset=np.linspace(-0.5, 0.5, 11),
    )

    def unnorm_lfunc(params, features, designs, **kwargs):
        y_mean = params.amplitude * np.sin(
            params.frequency * (designs.t_obs - params.offset)
        )
        y_diff = features.y_obs - y_mean
        return np.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)

    designer = ExperimentDesigner(
        params, features, designs, unnorm_lfunc,
        lfunc_args={"sigma_y": 0.1},
    )

    prior_amp = TopHat(np.linspace(0.5, 1.5, 11))
    prior_freq = TopHat(np.linspace(0.2, 2.0, 11))
    prior_off = TopHat(np.linspace(-0.5, 0.5, 11))
    prior = (
        prior_amp.reshape(-1, 1, 1)
        * prior_freq.reshape(1, -1, 1)
        * prior_off.reshape(1, 1, -1)
    )

    designer.calculateEIG(prior)

    return {
        "designer": designer,
        "prior": prior,
        "params": params,
        "features": features,
        "designs": designs,
    }
