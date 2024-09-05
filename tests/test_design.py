import unittest

import numpy as np

from bed.grid import Grid, GridStack
from bed.design import ExperimentDesigner


class TestDesign(unittest.TestCase):

    def test_sine_wave(self, sigma_y=0.1):
        designs = Grid(t_obs=np.linspace(0, 5, 51))
        features = Grid(y_obs=np.linspace(-1.25, 1.25, 100))
        params = Grid(amplitude=1, frequency=np.linspace(0.2, 2.0, 181), offset=0)

        with GridStack(features, designs, params):
            y_mean = params.amplitude * np.sin(
                params.frequency * (designs.t_obs - params.offset)
            )
            y_diff = features.y_obs - y_mean
            likelihood = np.exp(-0.5 * (y_diff / sigma_y) ** 2)
            features.normalize(likelihood)
        self.assertEqual(likelihood.shape, (100, 51, 1, 181, 1))
        self.assertAlmostEqual(likelihood.max(), 0.10114700589838203)

        designer = ExperimentDesigner(params, features, designs, likelihood)

        prior = np.ones(params.shape)
        params.normalize(prior)
        self.assertAlmostEqual(prior.sum(), 1)

        best = designer.calculateEIG(prior)

        self.assertEqual(best["t_obs"], 3.5)
        self.assertEqual(designer.posterior.shape, likelihood.shape)
        self.assertEqual(designer.marginal.shape, (100, 51))
        self.assertEqual(designer.IG.shape, (100, 51))
        self.assertEqual(designer.EIG.shape, (51,))

        self.assertAlmostEqual(designer.EIG.min(), 0)
        self.assertAlmostEqual(designer.EIG.max(), 2.4501367058730814)
