import unittest

import numpy as np

from bed.grid import Grid


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


if __name__ == "__main__":
    unittest.main()
