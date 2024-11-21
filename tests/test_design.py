import unittest

import numpy as np

from bed.grid import Grid, GridStack
from bed.design import ExperimentDesigner


class TestDesign(unittest.TestCase):

    def test_sine_wave(self, sigma_y=0.1):
        designs = Grid(t_obs=np.linspace(0, 5, 51))
        features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
        params = Grid(amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0)

        def unnorm_lfunc(params, features, designs, **kwargs):
            y_mean = params.amplitude * np.sin(
                params.frequency * (designs.t_obs - params.offset)
            )
            y_diff = features.y_obs - y_mean
            likelihood = np.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)
            return likelihood

        designer = ExperimentDesigner(params, 
            features, 
            designs, 
            unnorm_lfunc, 
            lfunc_args={'sigma_y': 0.1})

        prior = np.ones(params.shape)
        params.normalize(prior)
        self.assertAlmostEqual(prior.sum(), 1)

        best = designer.calculateEIG(prior)
        self.assertEqual(designer.subgrid_shape, (51,))
        self.assertEqual(best["t_obs"], 3.5)
        self.assertEqual(designer.marginal.shape, (100, 51))
        self.assertEqual(designer.IG.shape, (100, 51))
        self.assertEqual(designer.EIG.shape, (51,))

        self.assertAlmostEqual(designer.EIG.min(), 0)
        self.assertAlmostEqual(designer.EIG.max(), 2.4501367058730814)

    def test_sine_wave_subgrid(self):
        designs = Grid(t_obs=np.linspace(0, 5, 51))
        features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
        params = Grid(amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0)

        def unnorm_lfunc(params, features, designs, **kwargs):
            y_mean = params.amplitude * np.sin(
                params.frequency * (designs.t_obs - params.offset)
            )
            y_diff = features.y_obs - y_mean
            likelihood = np.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)
            return likelihood

        designer = ExperimentDesigner(params, 
            features, 
            designs, 
            unnorm_lfunc, 
            lfunc_args={'sigma_y': 0.1},
            mem=3)

        prior = np.ones(params.shape)
        params.normalize(prior)
        self.assertAlmostEqual(prior.sum(), 1)

        best = designer.calculateEIG(prior)
        self.assertEqual(designer.subgrid_shape, (10,))
        self.assertEqual(best["t_obs"], 3.5)
        self.assertEqual(designer.marginal.shape, (100, 10))
        self.assertEqual(designer.IG.shape, (100, 10))
        self.assertEqual(designer.EIG.shape, (51,))

        self.assertAlmostEqual(designer.EIG.min(), 0)
        self.assertAlmostEqual(designer.EIG.max(), 2.4501367058730814)


    def test_sine_wave_subgrid_invalid(self):
        designs = Grid(t_obs=np.linspace(0, 5, 51))
        features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
        params = Grid(amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0)

        def unnorm_lfunc(params, features, designs, **kwargs):
            y_mean = params.amplitude * np.sin(
                params.frequency * (designs.t_obs - params.offset)
            )
            y_diff = features.y_obs - y_mean
            likelihood = np.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)
            return likelihood
        with self.assertRaises(ValueError):
            designer = ExperimentDesigner(params, 
                features, 
                designs, 
                unnorm_lfunc, 
                lfunc_args={'sigma_y': 0.1},
                mem=0)