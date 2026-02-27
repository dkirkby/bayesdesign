"""Shared pytest fixtures for baseline compatibility tests."""

import numpy as np
import pytest


@pytest.fixture(
    scope="module",
    params=["numpy", "jax"],
    ids=["backend=numpy", "backend=jax"],
)
def backend(request):
    if request.param == "numpy":
        from bed.design import ExperimentDesigner
        from bed.grid import CosineBump, Gaussian, Grid, GridStack, TopHat

        return {
            "name": "numpy",
            "Grid": Grid,
            "GridStack": GridStack,
            "TopHat": TopHat,
            "CosineBump": CosineBump,
            "Gaussian": Gaussian,
            "ExperimentDesigner": ExperimentDesigner,
            "xp": np,
            "rtol": 1e-12,
            "atol": 1e-14,
        }

    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from bed_jax.design import ExperimentDesigner
        from bed_jax.grid import CosineBump, Gaussian, Grid, GridStack, TopHat
    except Exception as exc:
        pytest.skip(f"JAX backend unavailable: {exc}")

    return {
        "name": "jax",
        "Grid": Grid,
        "GridStack": GridStack,
        "TopHat": TopHat,
        "CosineBump": CosineBump,
        "Gaussian": Gaussian,
        "ExperimentDesigner": ExperimentDesigner,
        "xp": jnp,
        "rtol": 1e-7,
        "atol": 1e-10,
    }


def _sine_wave_lfunc(params, features, designs, xp, sigma_y):
    """Gaussian likelihood for sine wave model."""
    y_mean = params.amplitude * xp.sin(
        params.frequency * (designs.t_obs - params.offset)
    )
    y_diff = features.y_obs - y_mean
    return xp.exp(-0.5 * (y_diff / sigma_y) ** 2)


@pytest.fixture(scope="module")
def sine_wave_designer(backend):
    """Sine wave scenario: 1D frequency estimation with fixed amplitude and offset."""
    Grid = backend["Grid"]
    ExperimentDesigner = backend["ExperimentDesigner"]
    xp = backend["xp"]

    designs = Grid(t_obs=np.linspace(0, 5, 51))
    features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
    params = Grid(amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0)

    designer = ExperimentDesigner(
        params,
        features,
        designs,
        lambda p, f, d, **kwargs: _sine_wave_lfunc(
            p, f, d, xp, kwargs["sigma_y"]
        ),
        lfunc_args={"sigma_y": 0.1},
    )

    prior = np.ones(params.shape)
    prior = params.normalize(prior)
    designer.calculateEIG(prior)

    return {
        "designer": designer,
        "prior": prior,
        "params": params,
        "features": features,
        "designs": designs,
        "backend": backend["name"],
        "rtol": backend["rtol"],
        "atol": backend["atol"],
    }


@pytest.fixture(scope="module")
def sine_wave_designer_subgrid(backend):
    """Sine wave scenario with mem=3 (subgrid chunking)."""
    Grid = backend["Grid"]
    ExperimentDesigner = backend["ExperimentDesigner"]
    xp = backend["xp"]

    designs = Grid(t_obs=np.linspace(0, 5, 51))
    features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
    params = Grid(amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0)

    designer = ExperimentDesigner(
        params,
        features,
        designs,
        lambda p, f, d, **kwargs: _sine_wave_lfunc(
            p, f, d, xp, kwargs["sigma_y"]
        ),
        lfunc_args={"sigma_y": 0.1},
        mem=3,
    )

    prior = np.ones(params.shape)
    prior = params.normalize(prior)
    designer.calculateEIG(prior)

    return {
        "designer": designer,
        "prior": prior,
        "params": params,
        "features": features,
        "designs": designs,
        "backend": backend["name"],
        "rtol": backend["rtol"],
        "atol": backend["atol"],
    }


@pytest.fixture(scope="module")
def multi_param_designer(backend):
    """Multi-parameter scenario: 3D parameter grid (amplitude, frequency, offset)."""
    Grid = backend["Grid"]
    TopHat = backend["TopHat"]
    ExperimentDesigner = backend["ExperimentDesigner"]
    xp = backend["xp"]

    designs = Grid(t_obs=np.linspace(0, 4, 32))
    features = Grid(y_obs=np.linspace(-1.4, 1.4, 40))
    params = Grid(
        amplitude=np.linspace(0.5, 1.5, 11),
        frequency=np.linspace(0.2, 2.0, 11),
        offset=np.linspace(-0.5, 0.5, 11),
    )

    def unnorm_lfunc(params, features, designs, **kwargs):
        y_mean = params.amplitude * xp.sin(
            params.frequency * (designs.t_obs - params.offset)
        )
        y_diff = features.y_obs - y_mean
        return xp.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)

    designer = ExperimentDesigner(
        params,
        features,
        designs,
        unnorm_lfunc,
        lfunc_args={"sigma_y": 0.1},
    )

    prior_amp = TopHat(np.linspace(0.5, 1.5, 11))
    prior_freq = TopHat(np.linspace(0.2, 2.0, 11))
    prior_off = TopHat(np.linspace(-0.5, 0.5, 11))
    prior = (
        np.asarray(prior_amp).reshape(-1, 1, 1)
        * np.asarray(prior_freq).reshape(1, -1, 1)
        * np.asarray(prior_off).reshape(1, 1, -1)
    )

    designer.calculateEIG(prior)

    return {
        "designer": designer,
        "prior": prior,
        "params": params,
        "features": features,
        "designs": designs,
        "backend": backend["name"],
        "rtol": backend["rtol"],
        "atol": backend["atol"],
    }
