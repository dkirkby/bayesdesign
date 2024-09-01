"""Utility functions for working with discrete grids of variables."""

import inspect
import numpy as np


class Grid:

    def __init__(self, constraint=None, **axes):
        self.axes = {}
        naxes = len(axes)
        for offset, name in enumerate(axes):
            # Check for a valid axis definition
            axis = np.atleast_1d(axes[name])
            if axis.ndim != 1:
                raise RuntimeError(f'Invalid grid for axis "{name}"')
            # Store the axis offset to its position in the grid
            shape = [1] * offset + [-1] + [1] * (naxes - offset - 1)
            self.axes[name] = axis.reshape(shape)
        self.names = tuple(self.axes.keys())
        self.shape = list([axis.size for axis in self.axes.values()])
        self.expanded_shape = tuple(self.shape)
        # Check for a constraint function
        if constraint is not None:
            if not inspect.isfunction(constraint):
                raise ValueError("constraint must be a callable function")
            constraint_names = list(inspect.signature(constraint).parameters.keys())
            for name in constraint_names:
                if name not in self.names:
                    raise ValueError("constraint uses an invalid axis name: " + name)
            # Evaluate the constraint function on the full grid
            self.constraint_args = {name: self.axes[name] for name in constraint_names}
            constraint_eval = constraint(**self.constraint_args)
            if np.any(constraint_eval < 0):
                raise ValueError("constraint must be non-negative")
            self.constraint_weights = np.ones(self.shape, dtype=np.float32)
            self.constraint_weights[:] = constraint_eval
            # Replace the constrained axes with the reduced set of values
            mapper = dict(zip(constraint_names, np.squeeze(constraint_eval).nonzero()))
            first_offset = None
            for offset, name in enumerate(self.names):
                if name not in constraint_names:
                    continue
                if first_offset is None:
                    first_offset = offset
                shape = [1] * first_offset + [-1] + [1] * (naxes - first_offset - 1)
                self.axes[name] = self.axes[name].ravel()[mapper[name]].reshape(shape)
                self.shape[offset] = (
                    self.axes[name].size if offset == first_offset else 1
                )
        else:
            self.constraint_weights = 1
        self.shape = tuple(self.shape)
        # Initialize data used to implement GridStack
        self._stack_offset = 0
        self._stack_pad = 0

    def expand(self, values):
        """Expand an array of values to the full grid shape.
        Any values removed by a constraint will be set to NaN.
        """
        if values.shape != self.shape:
            raise ValueError(
                f"values shape {values.shape} does not match grid shape {self.shape}"
            )
        if self.expanded_shape == self.shape:
            return values
        expanded = np.full(self.expanded_shape, np.nan)
        indices = self.constraint_weights.nonzero()
        expanded[indices] = values.ravel()
        return expanded

    def __str__(self):
        return "[" + ",".join(self.names) + "]"

    def __repr__(self):
        return (
            "["
            + ", ".join(
                [
                    f"{axis.size}:{name}"
                    for (i, (name, axis)) in enumerate(self.axes.items())
                ]
            )
            + "]"
        )

    def __getattr__(self, name):
        """Return a 1D array of values for the named axis."""
        axis = self.axes[name]
        return axis.reshape(axis.shape + tuple([1] * self._stack_pad))

    def axis(self, name):
        """Return the index of the named axis. Axes in the same keep group will have the same index.
        Used to implement our index() method.
        """
        try:
            idx = self.names.index(name)
        except ValueError:
            raise ValueError(f'"{name}" is not in the grid')
        return idx + self._stack_offset

    def sum(self, values, keepdims=False, axis_names=None):
        """Sum values over our grid.
        Used for marginalization and to implement our normalize() method.
        Note that if axis_names uses a subset of a keep_group you will probably not get the result you want.
        TODO: detect and error on this condition.
        """
        # Use all axes by default
        axis_names = axis_names or self.names
        try:
            axes = tuple(
                [self.names.index(name) + self._stack_offset for name in axis_names]
            )
        except ValueError:
            raise ValueError(f"Invalid axis_names: {axis_names}")
        return np.sum(values, axis=axes, keepdims=keepdims)

    def normalize(self, values):
        """Normalize the array values in place over our grid."""
        norm = self.sum(values, keepdims=True)
        values /= norm
        return values

    def extent(self, name):
        """Return the (min,max) extent of the named axis.
        Useful to set plot axis limits.
        """
        axis = self.axes[name]
        return (axis.min(), axis.max())

    def index(self, name, value):
        """Return the index of the named axis and the location along that axis that is closest
        to the specified value.
        Used to implement GridStack.at()
        """
        axis = self.axis(name)
        deltas = value - self.axes[name]
        idx = np.argmin(np.abs(deltas))
        return (axis, idx)


class GridStack:
    """A context manager to allow grids to be temporarily stacked together to create a larger grid."""

    def __init__(self, *grids):
        self.grids = grids

    def __str__(self):
        return "[" + ",".join([str(grid) for grid in self.grids]) + "]"

    def __repr__(self):
        return "[" + ", ".join([repr(grid) for grid in self.grids]) + "]"

    def __enter__(self):
        offset = 0
        pad = 0
        offset = sum([len(grid.axes) for grid in self.grids])
        for grid in self.grids[::-1]:
            assert (
                grid._stack_offset == 0 and grid._stack_pad == 0
            ), "Is there a duplicated grid in the stack?"
            ngrid = len(grid.axes)
            grid._stack_pad = pad
            pad += ngrid
            offset -= ngrid
            grid._stack_offset = offset
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for grid in self.grids:
            grid._stack_offset = 0
            grid._stack_pad = 0

    def at(self, **coords):
        naxes = sum([len(grid.axes) for grid in self.grids])
        idx = [slice(None)] * naxes
        found = {name: False for name in coords}
        for grid in self.grids:
            for name, value in coords.items():
                if found[name] or (name not in grid.axes):
                    continue
                (axis, loc) = grid.index(name, value)
                found[name] = True
                idx[axis] = loc
        missing = [name for name in found if not found[name]]
        if missing:
            raise ValueError(f'Invalid grid name(s): {", ".join(missing)}')
        return tuple(idx)
