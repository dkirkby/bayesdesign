"""JAX-runtime tests for bed_jax.grid."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")

from bed_jax.grid import CosineBump, Gaussian, Grid, TopHat

RTOL = 1e-6


def _is_jax_array(values):
    try:
        return isinstance(values, jax.Array)
    except Exception:  # pragma: no cover
        return type(values).__module__.startswith("jax")


def _gpu_available():
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def test_tophat_jax_array():
    x = jnp.linspace(0.2, 2.0, 181)
    y = TopHat(x)
    assert _is_jax_array(y)
    np.testing.assert_allclose(np.asarray(y).sum(), 1.0, rtol=RTOL)


def test_cosinebump_jax_array():
    x = jnp.linspace(0.8, 1.2, 50)
    y = CosineBump(x)
    assert _is_jax_array(y)
    np.testing.assert_allclose(np.asarray(y).sum(), 1.0, rtol=RTOL)


def test_gaussian_jax_array():
    x = jnp.linspace(-3, 3, 101)
    y = Gaussian(x, 0, 1)
    assert _is_jax_array(y)
    np.testing.assert_allclose(np.asarray(y).sum(), 1.0, rtol=RTOL)


def test_grid_sum_with_jax_values():
    grid = Grid(x=np.arange(3), y=np.arange(4))
    values = jnp.ones(grid.shape)
    total = grid.sum(values)
    partial = grid.sum(values, axis_names=("x",))
    assert _is_jax_array(total)
    assert _is_jax_array(partial)
    np.testing.assert_allclose(np.asarray(total), 12.0, rtol=RTOL)
    np.testing.assert_allclose(np.asarray(partial), np.full((4,), 3.0), rtol=RTOL)


def test_grid_normalize_returns_jax_array():
    grid = Grid(x=np.arange(3))
    values = jnp.array([1.0, 2.0, 3.0])
    normed = grid.normalize(values)
    assert _is_jax_array(normed)
    np.testing.assert_allclose(np.asarray(normed).sum(), 1.0, rtol=RTOL)


@pytest.mark.gpu
@pytest.mark.skipif(
    not _gpu_available(),
    reason="No JAX GPU backend available.",
)
def test_tophat_runs_with_gpu_device():
    device = jax.devices("gpu")[0]
    x = jax.device_put(jnp.linspace(0.2, 2.0, 181), device=device)
    y = TopHat(x)
    np.testing.assert_allclose(np.asarray(y).sum(), 1.0, rtol=RTOL)
