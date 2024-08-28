import unittest

import numpy as np

from bed.grid import Grid, GridStack


class TestGrid(unittest.TestCase):

    def test_ctor_np(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        self.assertEqual(grid.names, ["x", "y"])
        self.assertEqual(grid.offsets, [0, 1])
        self.assertTrue(np.array_equal(grid.x, [[0], [1], [2]]))

    def test_ctor_list(self):
        grid = Grid(x=[0, 1, 2], y=[0, 1, 2, 3])
        self.assertEqual(grid.names, ["x", "y"])
        self.assertEqual(grid.offsets, [0, 1])
        self.assertTrue(np.array_equal(grid.x, [[0], [1], [2]]))

    def test_ctor_keep(self):
        grid = Grid(
            x=np.arange(3),
            y=np.arange(4),
            keep__x__y=lambda x, y: x + y < 3,
        )
        self.assertEqual(grid.names, ["x", "y"])
        self.assertEqual(grid.offsets, [0, 0])
        self.assertTrue(np.array_equal(grid.x, [[0], [0], [0], [1], [1], [2]]))
        self.assertTrue(
            np.array_equal(grid.y, np.array([[0], [1], [2], [0], [1], [0]]))
        )

    def test_sum(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        self.assertEqual(grid.sum(np.ones(grid.shape)), 12)

    def test_sum_partial(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        partial = grid.sum(np.ones(grid.shape), axis_names=("x"))
        self.assertTrue(np.array_equal(partial, np.full((4,), 3)))
        full = grid.sum(np.ones(grid.shape), axis_names=("x", "y"))
        self.assertEqual(full, 12)

    def test_sum_invalid(self):
        grid = Grid(x=np.arange(3), y=np.arange(4))
        with self.assertRaises(ValueError):
            grid.sum(np.ones(grid.shape), axis_names=("x", "z"))


class TestGridStack(unittest.TestCase):

    def test_basic(self):
        g1 = Grid(x=np.arange(3), y=np.arange(4))
        g2 = Grid(x=np.arange(2), y=np.arange(5))
        with GridStack(g1, g2) as stack:
            self.assertEqual(g1.x.shape, (3, 1, 1, 1))
            self.assertEqual(g1.y.shape, (4, 1, 1))
            self.assertEqual(g2.x.shape, (2, 1))
            self.assertEqual(g2.y.shape, (5,))

    def test_keep(self):
        g1 = Grid(x=np.arange(3), y=np.arange(4), keep__x__y=lambda x, y: x + y < 3)
        g2 = Grid(x=np.arange(2), y=np.arange(5))
        with GridStack(g1, g2) as stack:
            self.assertEqual(g1.x.shape, (6, 1, 1, 1))
            self.assertEqual(g1.y.shape, (6, 1, 1, 1))
            self.assertEqual(g2.x.shape, (2, 1))
            self.assertEqual(g2.y.shape, (5,))


if __name__ == "__main__":
    unittest.main()
