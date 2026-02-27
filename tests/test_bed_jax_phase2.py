"""Phase 3 smoke tests for the bed_jax scaffold transition."""

import inspect

import numpy as np
import pytest

pytest.importorskip("jax")

import bed_jax
from bed_jax.design import ExperimentDesigner
from bed_jax.grid import Grid


def _dummy_lfunc(params, features, designs, **kwargs):
    y_mean = params.p * designs.t
    y_diff = features.y - y_mean
    return np.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)


def test_imports_and_symbols():
    assert bed_jax is not None
    assert callable(Grid)
    assert callable(ExperimentDesigner)


def test_grid_signatures():
    init_sig = inspect.signature(Grid.__init__)
    assert list(init_sig.parameters.keys()) == [
        "self",
        "constraint",
        "full_shape",
        "axes",
    ]
    assert init_sig.parameters["axes"].kind == inspect.Parameter.VAR_KEYWORD

    sum_sig = inspect.signature(Grid.sum)
    assert list(sum_sig.parameters.keys()) == [
        "self",
        "values",
        "keepdims",
        "axis_names",
        "verbose",
    ]


def test_designer_signatures():
    init_sig = inspect.signature(ExperimentDesigner.__init__)
    assert list(init_sig.parameters.keys()) == [
        "self",
        "parameters",
        "features",
        "designs",
        "unnorm_lfunc",
        "lfunc_args",
        "mem",
    ]

    eig_sig = inspect.signature(ExperimentDesigner.calculateEIG)
    assert list(eig_sig.parameters.keys()) == ["self", "prior", "debug"]

    posterior_sig = inspect.signature(ExperimentDesigner.get_posterior)
    assert list(posterior_sig.parameters.keys()) == ["self", "design_and_features"]
    assert (
        posterior_sig.parameters["design_and_features"].kind
        == inspect.Parameter.VAR_KEYWORD
    )


def test_designer_structural_init_and_core_methods():
    params = Grid(p=np.array([0.0, 1.0]))
    features = Grid(y=np.array([0.0, 1.0]))
    designs = Grid(t=np.array([0.0, 1.0]))
    designer = ExperimentDesigner(
        params, features, designs, _dummy_lfunc, lfunc_args={"sigma_y": 0.25}
    )
    assert designer.parameters is params
    assert designer.features is features
    assert designer.designs is designs
    assert designer.EIG.shape == designs.shape

    prior = np.array([0.5, 0.5])
    best = designer.calculateEIG(prior)
    assert "t" in best
    assert designer._initialized
    assert designer.EIG.shape == (2,)
    assert np.isfinite(designer.EIG).all()

    posterior = designer.get_posterior(t=0.0, y=0.0)
    assert posterior.shape == params.shape
    np.testing.assert_allclose(np.asarray(params.sum(posterior)), 1.0, rtol=1e-6)

    updated_best = designer.update(t=0.0, y=0.0)
    assert "t" in updated_best
