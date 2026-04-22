"""Shared pytest fixtures for baseline compatibility tests."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest


def _jax_device_only_kw(backend: dict) -> dict:
    """Return ``{\"device\": ...}`` for JAX fixtures; empty dict for NumPy."""
    if backend.get("name") != "jax":
        return {}
    return {"device": backend["jax_device"]}


@contextlib.contextmanager
def _jax_prior_device_scope(backend: dict):
    """Place TopHat / prior linspace work on the same JAX device as the grids."""
    if backend.get("name") != "jax":
        yield
        return
    import jax

    with jax.default_device(jax.devices(backend["jax_device"])[0]):
        yield


@pytest.fixture(
    scope="module",
    params=[
        "numpy",
        pytest.param("jax-cpu", id="backend=jax-cpu"),
        pytest.param("jax-gpu", id="backend=jax-gpu"),
    ],
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

    jax_device = "gpu" if request.param == "jax-gpu" else "cpu"

    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        if jax_device == "gpu":
            try:
                if not jax.devices("gpu"):
                    pytest.skip("No JAX GPU device available for jax-gpu backend.")
            except RuntimeError:
                pytest.skip("No JAX GPU device available for jax-gpu backend.")
        import jax.numpy as jnp
        from bed_jax.design import ExperimentDesigner
        from bed_jax.grid import CosineBump, Gaussian, Grid, GridStack, TopHat
    except Exception as exc:
        pytest.skip(f"JAX backend unavailable: {exc}")

    return {
        "name": "jax",
        "jax_device": jax_device,
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
    dkw = _jax_device_only_kw(backend)

    designs = Grid(t_obs=xp.linspace(0, 5, 51), **dkw)
    features = Grid(y_obs=xp.linspace(-1.25, 1.25, 100), **dkw)
    params = Grid(
        amplitude=xp.asarray(1.0),
        frequency=xp.linspace(0.2, 2.0, 181),
        offset=xp.asarray(0.0),
        **dkw,
    )

    designer = ExperimentDesigner(
        params,
        features,
        designs,
        lambda p, f, d, **kwargs: _sine_wave_lfunc(
            p, f, d, xp, kwargs["sigma_y"]
        ),
        lfunc_args={"sigma_y": 0.1},
        **dkw,
    )

    prior = xp.ones(params.shape)
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
    dkw = _jax_device_only_kw(backend)

    designs = Grid(t_obs=xp.linspace(0, 5, 51), **dkw)
    features = Grid(y_obs=xp.linspace(-1.25, 1.25, 100), **dkw)
    params = Grid(
        amplitude=xp.asarray(1.0),
        frequency=xp.linspace(0.2, 2.0, 181),
        offset=xp.asarray(0.0),
        **dkw,
    )

    designer = ExperimentDesigner(
        params,
        features,
        designs,
        lambda p, f, d, **kwargs: _sine_wave_lfunc(
            p, f, d, xp, kwargs["sigma_y"]
        ),
        lfunc_args={"sigma_y": 0.1},
        mem=3,
        **dkw,
    )

    prior = xp.ones(params.shape)
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
    dkw = _jax_device_only_kw(backend)

    designs = Grid(t_obs=xp.linspace(0, 4, 32), **dkw)
    features = Grid(y_obs=xp.linspace(-1.4, 1.4, 40), **dkw)
    params = Grid(
        amplitude=xp.linspace(0.5, 1.5, 11),
        frequency=xp.linspace(0.2, 2.0, 11),
        offset=xp.linspace(-0.5, 0.5, 11),
        **dkw,
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
        **dkw,
    )

    with _jax_prior_device_scope(backend):
        prior_amp = TopHat(xp.linspace(0.5, 1.5, 11))
        prior_freq = TopHat(xp.linspace(0.2, 2.0, 11))
        prior_off = TopHat(xp.linspace(-0.5, 0.5, 11))
    prior = (
        xp.asarray(prior_amp).reshape(-1, 1, 1)
        * xp.asarray(prior_freq).reshape(1, -1, 1)
        * xp.asarray(prior_off).reshape(1, 1, -1)
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
