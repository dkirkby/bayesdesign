"""JAX runtime and device-selection tests for bed_jax."""

import inspect

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")

import bed_jax
from bed_jax.design import ExperimentDesigner
from bed_jax.grid import CosineBump, Gaussian, Grid, TopHat

RTOL = 1e-6


def _is_jax_array(values):
    try:
        return isinstance(values, jax.Array)
    except Exception:  # pragma: no cover
        return type(values).__module__.startswith("jax")


@pytest.fixture(params=["cpu", "gpu"], ids=["device=cpu", "device=gpu"])
def target_device(request):
    try:
        devices = jax.devices(request.param)
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip(f"No {request.param} device available.")
    return devices[0]


def _dummy_lfunc(params, features, designs, **kwargs):
    y_mean = params.p * designs.t
    y_diff = features.y - y_mean
    return jnp.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)


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
        "device",
    ]

    eig_sig = inspect.signature(ExperimentDesigner.calculateEIG)
    assert list(eig_sig.parameters.keys()) == ["self", "prior", "debug"]

    posterior_sig = inspect.signature(ExperimentDesigner.get_posterior)
    assert list(posterior_sig.parameters.keys()) == ["self", "design_and_features"]
    assert (
        posterior_sig.parameters["design_and_features"].kind
        == inspect.Parameter.VAR_KEYWORD
    )


def test_tophat_jax_array():
    x = jnp.linspace(0.2, 2.0, 181)
    y = TopHat(x)
    assert _is_jax_array(y)
    assert bool(jnp.isclose(jnp.sum(y), 1.0, rtol=RTOL))


def test_cosinebump_jax_array():
    x = jnp.linspace(0.8, 1.2, 50)
    y = CosineBump(x)
    assert _is_jax_array(y)
    assert bool(jnp.isclose(jnp.sum(y), 1.0, rtol=RTOL))


def test_gaussian_jax_array():
    x = jnp.linspace(-3, 3, 101)
    y = Gaussian(x, 0, 1)
    assert _is_jax_array(y)
    assert bool(jnp.isclose(jnp.sum(y), 1.0, rtol=RTOL))


def test_grid_sum_with_jax_values():
    grid = Grid(x=jnp.arange(3), y=jnp.arange(4))
    values = jnp.ones(grid.shape)
    total = grid.sum(values)
    partial = grid.sum(values, axis_names=("x",))
    assert _is_jax_array(total)
    assert _is_jax_array(partial)
    assert bool(jnp.isclose(total, 12.0, rtol=RTOL))
    assert bool(jnp.allclose(partial, jnp.full((4,), 3.0), rtol=RTOL))


def test_grid_normalize_returns_jax_array():
    grid = Grid(x=jnp.arange(3))
    values = jnp.array([1.0, 2.0, 3.0])
    normed = grid.normalize(values)
    assert _is_jax_array(normed)
    assert bool(jnp.isclose(jnp.sum(normed), 1.0, rtol=RTOL))


def test_designer_structural_init_and_core_methods():
    params = Grid(p=jnp.array([0.0, 1.0]))
    features = Grid(y=jnp.array([0.0, 1.0]))
    designs = Grid(t=jnp.array([0.0, 1.0]))
    designer = ExperimentDesigner(
        params, features, designs, _dummy_lfunc, lfunc_args={"sigma_y": 0.25}
    )
    assert designer.parameters is params
    assert designer.features is features
    assert designer.designs is designs
    assert designer.EIG.shape == designs.shape

    prior = jnp.array([0.5, 0.5])
    best = designer.calculateEIG(prior)
    assert "t" in best
    assert designer._initialized
    assert designer.EIG.shape == (2,)
    assert bool(jnp.isfinite(designer.EIG).all())

    posterior = designer.get_posterior(t=0.0, y=0.0)
    assert posterior.shape == params.shape
    assert bool(jnp.isclose(params.sum(posterior), 1.0, rtol=1e-6))

    updated_best = designer.update(t=0.0, y=0.0)
    assert "t" in updated_best


def test_grid_place_on_device(target_device):
    grid = Grid(
        x=jnp.arange(3),
        y=jnp.arange(4),
        constraint=lambda x, y: x + y < 3,
    )
    grid._place_on_device(target_device)
    assert grid.x.device.platform == target_device.platform
    assert grid.y.device.platform == target_device.platform
    assert grid.constraint_eval.device.platform == target_device.platform


def test_designer_methods_run_on_selected_device(target_device):
    params = Grid(p=jnp.array([0.0, 1.0]), q=jnp.array([0.0, 1.0]))
    features = Grid(y=jnp.array([0.0, 1.0]))
    designs = Grid(t=jnp.array([0.0, 1.0]))
    designer = ExperimentDesigner(
        params,
        features,
        designs,
        _dummy_lfunc,
        lfunc_args={"sigma_y": 0.25},
        device=target_device.platform,
    )
    prior = jnp.ones(params.shape)
    prior = params.normalize(prior)
    best = designer.calculateEIG(prior)
    assert "t" in best
    assert designer.EIG.device.platform == target_device.platform

    marginal = designer.calculateMarginalEIG("q")
    assert marginal.device.platform == target_device.platform

    posterior = designer.get_posterior(t=0.0, y=0.0)
    assert posterior.device.platform == target_device.platform

    updated_best = designer.update(t=0.0, y=0.0)
    assert "t" in updated_best
    assert designer.EIG.device.platform == target_device.platform


def test_designer_invalid_device_selection():
    params = Grid(p=jnp.array([0.0, 1.0]))
    features = Grid(y=jnp.array([0.0, 1.0]))
    designs = Grid(t=jnp.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="device must be"):
        ExperimentDesigner(
            params,
            features,
            designs,
            _dummy_lfunc,
            lfunc_args={"sigma_y": 0.25},
            device="tpu",
        )


def test_designer_gpu_requested_without_gpu_errors():
    if any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("GPU backend available; unavailable-GPU path not applicable.")

    params = Grid(p=jnp.array([0.0, 1.0]))
    features = Grid(y=jnp.array([0.0, 1.0]))
    designs = Grid(t=jnp.array([0.0, 1.0]))
    with pytest.raises(RuntimeError, match="GPU device requested"):
        ExperimentDesigner(
            params,
            features,
            designs,
            _dummy_lfunc,
            lfunc_args={"sigma_y": 0.25},
            device="gpu",
        )


def test_tophat_runs_on_selected_device(target_device):
    x = jax.device_put(jnp.linspace(0.2, 2.0, 181), device=target_device)
    y = TopHat(x)
    assert y.device.platform == target_device.platform
    assert bool(jnp.isclose(jnp.sum(y), 1.0, rtol=RTOL))
