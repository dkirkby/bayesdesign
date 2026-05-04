import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from bed.grid import Grid, GridStack
from bed.design import ExperimentDesigner


@pytest.fixture(params=[1, 5, 24, 100], ids=lambda n: f"chunk={n}")
def design_chunk_size(request):
    return request.param


def _sine_lfunc_factory(xp):
    def unnorm_lfunc(params, features, designs, **kwargs):
        y_mean = params.amplitude * xp.sin(
            params.frequency * (designs.t_obs - params.offset)
        )
        y_diff = features.y_obs - y_mean
        return xp.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)

    return unnorm_lfunc


def test_design_chunk_size_parity(design_chunk_size):
    xp = jnp
    designs = Grid(t_obs=xp.linspace(0.0, 4.0, 24))
    features = Grid(y_obs=xp.linspace(-1.4, 1.4, 32))
    params = Grid(
        amplitude=xp.linspace(0.5, 1.5, 7),
        frequency=xp.linspace(0.2, 2.0, 9),
        offset=xp.linspace(-0.5, 0.5, 7),
    )

    unnorm_lfunc = _sine_lfunc_factory(xp)
    prior = params.normalize(xp.ones(params.shape))

    full_designer = ExperimentDesigner(
        params,
        features,
        designs,
        unnorm_lfunc,
        lfunc_args={"sigma_y": 0.1},
    )
    chunked_designer = ExperimentDesigner(
        params,
        features,
        designs,
        unnorm_lfunc,
        lfunc_args={"sigma_y": 0.1},
        design_chunk_size=design_chunk_size,
    )

    full_designer.calculateEIG(prior)
    chunked_designer.calculateEIG(prior)

    total_designs = int(np.prod(designs.shape))
    expected_subgrid = min(int(design_chunk_size), total_designs)
    expected_num_subgrids = np.ceil(total_designs / expected_subgrid)
    assert chunked_designer.design_subgrid == expected_subgrid
    assert chunked_designer.num_subgrids == expected_num_subgrids
    assert chunked_designer.subgrid_shape == (expected_subgrid,)

    assert np.allclose(
        np.asarray(chunked_designer.EIG),
        np.asarray(full_designer.EIG),
        rtol=1e-6,
        atol=1e-9,
    )


def test_design_chunk_size_invalid_or_conflicting():
    xp = jnp
    designs = Grid(t_obs=xp.array([0.0, 1.0]))
    features = Grid(y_obs=xp.array([0.0, 1.0]))
    params = Grid(
        amplitude=xp.array([1.0]),
        frequency=xp.array([1.0]),
        offset=xp.array([0.0]),
    )

    unnorm_lfunc = _sine_lfunc_factory(xp)

    with pytest.raises(ValueError, match="Design chunk size must be positive"):
        ExperimentDesigner(
            params,
            features,
            designs,
            unnorm_lfunc,
            lfunc_args={"sigma_y": 0.1},
            design_chunk_size=0,
        )

    with pytest.raises(ValueError, match="Specify at most one of mem or design_chunk_size"):
        ExperimentDesigner(
            params,
            features,
            designs,
            unnorm_lfunc,
            lfunc_args={"sigma_y": 0.1},
            mem=1,
            design_chunk_size=1,
        )


class TestDesign(unittest.TestCase):
    def test_sine_wave(self, sigma_y=0.1):
        designs = Grid(t_obs=jnp.linspace(0, 5, 51))
        features = Grid(y_obs=jnp.linspace(-1.25, 1.25, 100))
        params = Grid(amplitude=jnp.asarray(1), frequency=jnp.linspace(0.2, 2.0, 181), offset=jnp.asarray(0))

        def unnorm_lfunc(params, features, designs, **kwargs):
            y_mean = params.amplitude * jnp.sin(
                params.frequency * (designs.t_obs - params.offset)
            )
            y_diff = features.y_obs - y_mean
            likelihood = jnp.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)
            return likelihood

        designer = ExperimentDesigner(params, 
            features, 
            designs, 
            unnorm_lfunc, 
            lfunc_args={'sigma_y': 0.1})

        prior = params.normalize(jnp.ones(params.shape))
        self.assertAlmostEqual(float(jnp.sum(prior)), 1)

        best = designer.calculateEIG(prior)
        self.assertEqual(designer.subgrid_shape, (51,))
        self.assertEqual(best["t_obs"], 3.5)
        self.assertEqual(designer.marginal.shape, (100, 51))
        self.assertAlmostEqual(float(designer.get_posterior(t_obs=2.0, y_obs=0.2).max()), 0.07484451100061842)
        self.assertEqual(designer.IG.shape, (100, 51))
        self.assertEqual(designer.EIG.shape, (51,))

        self.assertAlmostEqual(float(designer.EIG.min()), 0)
        self.assertAlmostEqual(float(designer.EIG.max()), 2.4501367058730814)

    def test_sine_wave_subgrid(self):
        designs = Grid(t_obs=jnp.linspace(0, 5, 51))
        features = Grid(y_obs=jnp.linspace(-1.25, 1.25, 100))
        params = Grid(amplitude=jnp.asarray(1), frequency=jnp.linspace(0.2, 2.0, 181), offset=jnp.asarray(0))

        def unnorm_lfunc(params, features, designs, **kwargs):
            y_mean = params.amplitude * jnp.sin(
                params.frequency * (designs.t_obs - params.offset)
            )
            y_diff = features.y_obs - y_mean
            likelihood = jnp.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)
            return likelihood

        designer = ExperimentDesigner(params, 
            features, 
            designs, 
            unnorm_lfunc, 
            lfunc_args={'sigma_y': 0.1},
            mem=3)

        prior = params.normalize(jnp.ones(params.shape))
        self.assertAlmostEqual(float(jnp.sum(prior)), 1)

        best = designer.calculateEIG(prior)
        self.assertEqual(designer.subgrid_shape, (10,))
        self.assertEqual(best["t_obs"], 3.5)
        self.assertEqual(designer.marginal.shape, (100, 10))
        self.assertAlmostEqual(float(designer.get_posterior(t_obs=2.0, y_obs=0.2).max()), 0.07484451100061842)
        self.assertEqual(designer.IG.shape, (100, 10))
        self.assertEqual(designer.EIG.shape, (51,))

        self.assertAlmostEqual(float(designer.EIG.min()), 0)
        self.assertAlmostEqual(float(designer.EIG.max()), 2.4501367058730814)


    def test_sine_wave_subgrid_invalid(self):
        designs = Grid(t_obs=jnp.linspace(0, 5, 51))
        features = Grid(y_obs=jnp.linspace(-1.25, 1.25, 100))
        params = Grid(amplitude=jnp.asarray(1), frequency=jnp.linspace(0.2, 2.0, 181), offset=jnp.asarray(0))

        def unnorm_lfunc(params, features, designs, **kwargs):
            y_mean = params.amplitude * jnp.sin(
                params.frequency * (designs.t_obs - params.offset)
            )
            y_diff = features.y_obs - y_mean
            likelihood = jnp.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)
            return likelihood
        with self.assertRaises(ValueError):
            designer = ExperimentDesigner(params, 
                features, 
                designs, 
                unnorm_lfunc, 
                lfunc_args={'sigma_y': 0.1},
                mem=0)
