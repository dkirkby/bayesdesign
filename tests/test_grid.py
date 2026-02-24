import unittest

import numpy as np

from bed.grid import Grid, GridStack, PermutationInvariant, CosineBump


class TestGrid(unittest.TestCase):

    def test_ctor_np(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        self.assertEqual(grid.names, ("x", "y"))
        self.assertTrue(np.array_equal(grid.x, [[0], [1], [2]]))

    def test_ctor_list(self):
        grid = Grid(x=[0, 1, 2], y=[0, 1, 2, 3])
        self.assertEqual(grid.names, ("x", "y"))
        self.assertTrue(np.array_equal(grid.x, [[0], [1], [2]]))

    def test_ctor_constraint(self):
        grid = Grid(
            x=np.arange(3),
            y=np.arange(4),
            constraint=lambda x, y: x + y < 3,
        )
        self.assertEqual(grid.names, ("x", "y"))
        self.assertTrue(np.array_equal(grid.x, [[0], [0], [0], [1], [1], [2]]))
        self.assertTrue(
            np.array_equal(grid.y, np.array([[0], [1], [2], [0], [1], [0]]))
        )
        self.assertTrue(np.array_equal(grid.axes_in["y"], np.arange(4)))

    def test_ctor_constraint_invalid(self):
        with self.assertRaises(ValueError):
            Grid(x=[1, 2, 3], y=[4, 5], z=[6, 7, 8], constraint=lambda a, b: a < b)

    def test_expand_2d(self):
        g1 = Grid(x=[1, 2, 3], u=[4, 5], y=1, v=[6, 7, 8, 9])
        g2 = Grid(
            x=[1, 2, 3],
            u=[4, 5],
            y=1,
            v=[6, 7, 8, 9],
            constraint=lambda u, v: (u + v) % 2,
        )
        z1 = g1.x * ((g1.u + g1.v) % 2) ** g1.y
        z2 = g2.x * ((g2.u + g2.v) % 2) ** g2.y
        z1e = g1.expand(z1)
        z2e = g2.expand(z2)
        assert z1e is z1
        nonzero = ~np.isnan(z2e)
        assert np.array_equal(z1[nonzero], z2e[nonzero])

    def test_expand_3d(self):
        g1 = Grid(x=[1, 2, 3], u=2, y=[1, 2, 3], v=[1, 2], z=[1, 2, 3])
        z1 = (g1.x + g1.y + g1.z) * g1.u + g1.v
        g2 = Grid(
            x=[1, 2, 3],
            u=2,
            y=[1, 2, 3],
            v=[1, 2],
            z=[1, 2, 3],
            constraint=lambda x, y, z: PermutationInvariant(x, y, z),
        )
        z2 = (g2.x + g2.y + g2.z) * g2.u + g2.v
        z1e = g1.expand(z1)
        z2e = g2.expand(z2)
        assert z1e is z1
        nonzero = ~np.isnan(z2e)
        assert np.array_equal(z1[nonzero], z2e[nonzero])

    def test_sum(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        self.assertEqual(grid.sum(np.ones(grid.shape)), 12)

    def test_sum_partial(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        partial = grid.sum(np.ones(grid.shape), axis_names=("x"))
        self.assertTrue(np.array_equal(partial, np.full((4,), 3)))
        full = grid.sum(np.ones(grid.shape), axis_names=("x", "y"))
        self.assertEqual(full, 12)

    def test_sum_constraint(self):
        g1 = Grid(x=[1, 2, 3], u=[4, 5], y=1, v=[6, 7, 8, 9])
        g2 = Grid(
            x=[1, 2, 3],
            u=[4, 5],
            y=1,
            v=[6, 7, 8, 9],
            constraint=lambda u, v: (u + v) % 2 == 1,
        )
        z1 = g1.x * ((g1.u + g1.v) % 2) ** g1.y
        z2 = g2.x * ((g2.u + g2.v) % 2) ** g2.y
        self.assertEqual(g1.sum(z1), g2.sum(z2))

    def test_sum_permutation_invariant_2d(self):
        g1 = Grid(u=2, x=[1, 2, 3], y=[1, 2], z=[1, 2, 3])
        z1 = (g1.x + g1.z) * g1.y + g1.u
        g2 = Grid(
            u=2,
            x=[1, 2, 3],
            y=[1, 2],
            z=[1, 2, 3],
            constraint=lambda x, z: PermutationInvariant(x, z),
        )
        z2 = (g2.x + g2.z) * g2.y + g2.u
        self.assertEqual(g1.sum(z1), g2.sum(z2))

    def test_sum_permutation_invariant_3d(self):
        g1 = Grid(u=[1, 2], x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3])
        z1 = (g1.x + g1.y + g1.z) * g1.u
        g2 = Grid(
            u=[1, 2],
            x=[1, 2, 3],
            y=[1, 2, 3],
            z=[1, 2, 3],
            constraint=lambda x, y, z: PermutationInvariant(x, y, z),
        )
        z2 = (g2.x + g2.y + g2.z) * g2.u
        self.assertEqual(g1.sum(z1), g2.sum(z2))

    def test_sum_marginalized_with_axis_names(self):
        """Test that Grid.sum accepts marginalized values when axis_names specifies the non-marginalized axes."""
        grid = Grid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
        values = np.ones(grid.shape)
        # Marginalize over x and z, keeping dims.
        marginal = grid.sum(values, axis_names=("x", "z"), keepdims=True)
        self.assertEqual(marginal.shape, (1, 4, 1))
        np.testing.assert_array_equal(marginal, np.full((1, 4, 1), 15))
        # Sum the marginalized result over the remaining (interest) axis.
        total = grid.sum(marginal, axis_names=("y",))
        self.assertEqual(total, 60)

    def test_sum_marginalized_in_stack(self):
        """Test that Grid.sum accepts marginalized values with axis_names inside a GridStack."""
        g1 = Grid(a=np.arange(2))
        g2 = Grid(x=np.arange(3), y=np.arange(4))
        with GridStack(g1, g2):
            values = np.ones(g1.shape + g2.shape)  # (2, 3, 4)
            # Marginalize g2 over x, keeping dims.
            marginal = g2.sum(values, axis_names=("x",), keepdims=True)
            self.assertEqual(marginal.shape, (2, 1, 4))
            # Sum the marginalized result over the remaining axis.
            total = g2.sum(marginal, axis_names=("y",))
            self.assertEqual(total.shape, (2,))
            np.testing.assert_array_equal(total, [12, 12])

    def test_sum_marginalized_rejects_without_axis_names(self):
        """Test that Grid.sum rejects marginalized values when axis_names is not specified."""
        grid = Grid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
        marginal = grid.sum(np.ones(grid.shape), axis_names=("x", "z"), keepdims=True)
        # Without axis_names, the full shape check is strict.
        with self.assertRaises(ValueError):
            grid.sum(marginal)

    def test_sum_marginalized_rejects_bad_named_axis(self):
        """Test that Grid.sum rejects marginalized values when a named axis has been marginalized."""
        grid = Grid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
        marginal = grid.sum(np.ones(grid.shape), axis_names=("x",), keepdims=True)
        # Trying to sum over x (which is now size 1) should fail.
        with self.assertRaises(ValueError):
            grid.sum(marginal, axis_names=("x",))

    def test_sum_marginalized_rejects_wrong_ndim(self):
        """Test that Grid.sum rejects values with wrong number of dimensions."""
        grid = Grid(x=np.arange(3), y=np.arange(4))
        with self.assertRaises(ValueError):
            grid.sum(np.ones((3,)), axis_names=("x",))

    def test_sum_invalid(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        with self.assertRaises(ValueError):
            grid.sum(np.ones(grid.shape), axis_names=("x", "z"))

    def test_subgrid(self):
        grid = Grid(x=np.arange(3), y=np.arange(3))
        s, mask = next(grid.subgrid(2))
        self.assertTrue(np.array_equal(s.x, [[0], [0]]))
        self.assertTrue(np.array_equal(s.y, [[0], [1]]))
        self.assertTrue(len(list(grid.subgrid(2))) == 5)
        self.assertTrue(np.array_equal(mask, 
        [[ True,  True, False],
        [False, False, False],
        [False, False, False]]))

    def test_subgrid_constrained(self):
        grid = Grid(x=np.arange(3), y=np.arange(3), constraint=lambda x, y: x + y < 3)
        s, mask = next(grid.subgrid(3))
        self.assertTrue(np.array_equal(s.x, [[0], [0], [0]]))
        self.assertTrue(np.array_equal(s.y, [[0], [1], [2]]))
        self.assertTrue(len(list(grid.subgrid(3))) == 2)
        self.assertTrue(np.array_equal(mask, [[True],[True],[True],[False],[False],[False]]))

    def test_subgrid_invalid(self):
        grid = Grid(x=np.arange(3), y=np.arange(3))
        with self.assertRaises(ValueError):
            next(grid.subgrid(2.1))
        with self.assertRaises(ValueError):
            next(grid.subgrid(-1))

class TestPermutationInvariant(unittest.TestCase):

    def test_basic_2d(self):
        x = np.arange(3)
        pi = PermutationInvariant(x, x.reshape((-1, 1)))
        self.assertTrue(np.array_equal(pi, np.array([[1, 2, 2], [0, 1, 2], [0, 0, 1]])))

    def test_basic_3d(self):
        x = np.arange(3)
        pi = PermutationInvariant(x.reshape((-1, 1, 1)), x.reshape((-1, 1)), x)
        self.assertTrue(
            np.array_equal(
                pi,
                np.array(
                    [
                        [[1.0, 3.0, 3.0], [0.0, 3.0, 6.0], [0.0, 0.0, 3.0]],
                        [[0.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 3.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    ]
                ),
            )
        )

    def test_invalid_axes(self):
        with self.assertRaises(ValueError):
            PermutationInvariant(np.array([1, 2, 3]), np.array([1, 2]))


class TestCosineBump(unittest.TestCase):

    def test_basic(self):
        y = CosineBump(np.linspace(-4, 7, 100).reshape(1, -1, 1)).squeeze()
        self.assertEqual(y[0], 0)
        self.assertEqual(y[-1], 0)
        self.assertTrue(np.allclose(np.sum(y), 1))


class TestGridStack(unittest.TestCase):

    def test_basic(self):
        g1 = Grid(x=np.arange(3), y=np.arange(4))
        g2 = Grid(x=np.arange(2), y=np.arange(5))
        with GridStack(g1, g2) as stack:
            self.assertEqual(g1.x.shape, (3, 1, 1, 1))
            self.assertEqual(g1.y.shape, (1, 4, 1, 1))
            self.assertEqual(g2.x.shape, (2, 1))
            self.assertEqual(g2.y.shape, (1, 5))

    def test_keep(self):
        g1 = Grid(x=np.arange(3), y=np.arange(4), constraint=lambda x, y: x + y < 3)
        g2 = Grid(x=np.arange(2), y=np.arange(5))
        with GridStack(g1, g2) as stack:
            self.assertEqual(g1.x.shape, (6, 1, 1, 1))
            self.assertEqual(g1.y.shape, (6, 1, 1, 1))
            self.assertEqual(g2.x.shape, (2, 1))
            self.assertEqual(g2.y.shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
