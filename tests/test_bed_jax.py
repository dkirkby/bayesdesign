"""JAX runtime and device-selection tests for bed_jax."""

import inspect

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")

import bed_jax
from bed_jax.design import ExperimentDesigner
from bed_jax.grid import CosineBump, Gaussian, Grid, PermutationInvariant, TopHat

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


def _sine_lfunc(params, features, designs, **kwargs):
    y_mean = params.amplitude * jnp.sin(
        params.frequency * (designs.t_obs - params.offset)
    )
    y_diff = features.y_obs - y_mean
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
        "device",
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
        "design_chunk_size",
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


def test_grid_device_placement(target_device):
    grid = Grid(
        device=target_device.platform,
        x=jnp.array([1.0, 2.0, 3.0]),
        y=jnp.array([4.0, 5.0]),
    )
    assert grid.device.platform == target_device.platform
    for name in grid.names:
        assert grid.axes_in[name].device.platform == target_device.platform

    # Operations should preserve device placement.
    with jax.default_device(target_device):
        values = jnp.ones(grid.shape)
    normed = grid.normalize(values)
    assert normed.device.platform == target_device.platform

    summed = grid.sum(values)
    assert summed.device.platform == target_device.platform


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


def test_designer_methods_run_on_selected_device(target_device):
    params = Grid(
        device=target_device.platform,
        p=jnp.array([0.0, 1.0]),
        q=jnp.array([0.0, 1.0]),
    )
    features = Grid(device=target_device.platform, y=jnp.array([0.0, 1.0]))
    designs = Grid(device=target_device.platform, t=jnp.array([0.0, 1.0]))
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


def test_designer_repeated_calculateEIG_stable_and_cached():
    params = Grid(p=jnp.array([0.0, 1.0, 2.0]))
    features = Grid(y=jnp.linspace(-1.0, 1.0, 21))
    designs = Grid(t=jnp.linspace(0.0, 1.0, 11))
    designer = ExperimentDesigner(
        params, features, designs, _dummy_lfunc, lfunc_args={"sigma_y": 0.25}
    )
    prior = params.normalize(jnp.ones(params.shape))

    best1 = designer.calculateEIG(prior)
    eig1 = jnp.array(designer.EIG)
    posterior = designer.get_posterior(t=0.5, y=0.2)

    best2 = designer.calculateEIG(prior)
    eig2 = jnp.array(designer.EIG)
    assert best1.keys() == best2.keys()
    assert bool(jnp.allclose(eig1, eig2, rtol=RTOL))
    assert posterior.shape == params.shape


def test_designer_subgrid_fullgrid_parity():
    designs = Grid(t_obs=jnp.linspace(0, 4, 24))
    features = Grid(y_obs=jnp.linspace(-1.4, 1.4, 32))
    params = Grid(
        amplitude=jnp.linspace(0.5, 1.5, 7),
        frequency=jnp.linspace(0.2, 2.0, 9),
        offset=jnp.linspace(-0.5, 0.5, 7),
    )

    full_designer = ExperimentDesigner(
        params, features, designs, _sine_lfunc, lfunc_args={"sigma_y": 0.1}
    )
    subgrid_designer = ExperimentDesigner(
        params, features, designs, _sine_lfunc, lfunc_args={"sigma_y": 0.1}, mem=2
    )

    prior = params.normalize(jnp.ones(params.shape))
    full_designer.calculateEIG(prior)
    subgrid_designer.calculateEIG(prior)
    assert bool(
        jnp.allclose(full_designer.EIG, subgrid_designer.EIG, rtol=1e-6, atol=1e-9)
    )

    full_marginal = full_designer.calculateMarginalEIG("amplitude", "offset")
    subgrid_marginal = subgrid_designer.calculateMarginalEIG("amplitude", "offset")
    assert bool(
        jnp.allclose(full_marginal, subgrid_marginal, rtol=1e-6, atol=1e-9)
    )


def test_designer_requested_device_mismatch_raises():
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if not gpu_devices:
        pytest.skip("No GPU backend available for mismatch test.")

    params = Grid(device="cpu", p=jnp.array([0.0, 1.0]))
    features = Grid(device="cpu", y=jnp.array([0.0, 1.0]))
    designs = Grid(device="cpu", t=jnp.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="does not match ExperimentDesigner device"):
        ExperimentDesigner(
            params,
            features,
            designs,
            _dummy_lfunc,
            lfunc_args={"sigma_y": 0.25},
            device="gpu",
        )


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
    with jax.default_device(target_device):
        x = jnp.linspace(0.2, 2.0, 181)
    y = TopHat(x)
    assert y.device.platform == target_device.platform
    assert bool(jnp.isclose(jnp.sum(y), 1.0, rtol=RTOL))
