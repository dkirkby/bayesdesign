"""Utility functions for working with discrete grids of variables."""

import inspect
import math

import numpy as np


class Grid:

    def __init__(self, constraint=None, full_shape=None, **axes):
        self.axes = {}
        self.axes_in = {}
        naxes = len(axes)
        for offset, name in enumerate(axes):
            # Check for a valid axis definition
            axis = np.atleast_1d(axes[name])
            # Store the axis offset to its position in the grid
            shape = [1] * offset + [-1] + [1] * (naxes - offset - 1)
            self.axes[name] = axis.reshape(shape)
            # Remember the original axis definition
            self.axes_in[name] = axis
        self.names = tuple(self.axes.keys())
        self.shape = list([axis.size for axis in self.axes.values()])
        self.expanded_shape = tuple(self.shape)
        self.constraint = constraint
        # Check for a constraint function
        if constraint is not None:
            if not inspect.isfunction(constraint):
                raise ValueError("constraint must be a callable function")
            constraint_names = list(inspect.signature(constraint).parameters.keys())
            if "idx" in constraint_names:
                constraint_names = list(self.names) + ["idx"]
            elif "kwargs" in constraint_names:
                constraint_names = list(self.names)
            for name in constraint_names:
                if name not in self.names and name != "idx" and name != "kwargs":
                    raise ValueError("constraint uses an invalid axis name: " + name)
            # Evaluate the constraint function on the full grid
            self.constraint_args = {name: self.axes[name] for name in constraint_names if name != "idx"}
            if "idx" in constraint_names:
                self.constraint_args["idx"] = np.arange(np.prod(full_shape)).reshape(full_shape)
            constraint_eval = constraint(**self.constraint_args)
            self.constraint_eval = constraint_eval
            if np.any(constraint_eval < 0):
                raise ValueError("constraint must be non-negative")
            # Replace the constrained axes with the reduced set of values
            if "idx" in constraint_names:
                mapper = dict(zip(constraint_names, np.squeeze(constraint_eval).nonzero() * np.ones((len(self.names), 1)).astype(int)))
            else:
                mapper = dict(zip(constraint_names, np.squeeze(constraint_eval).nonzero()))
            constraint_offsets = []
            for offset, name in enumerate(self.names):
                if name not in constraint_names:
                    continue
                if len(constraint_offsets) == 0:
                    shape = [1] * offset + [-1] + [1] * (naxes - offset - 1)
                    self.shape[offset] = np.count_nonzero(constraint_eval)
                else:
                    self.shape[offset] = 1
                self.axes[name] = self.axes[name].ravel()[mapper[name]].reshape(shape)
                constraint_offsets.append(offset)
            first_offset = constraint_offsets[0]
            shape = tuple([1] * first_offset + [-1] + [1] * (naxes - first_offset - 1))
            mask = constraint_eval != 0
            self.constraint_weights = constraint_eval[mask].ravel().reshape(shape)
            # Compute and save the indices needed to expand an array tabulated on the reduced grid
            self.constraint_offsets = np.array(constraint_offsets)
            self.constraint_indices = list(constraint_eval.nonzero())
            for offset in range(naxes):
                if offset not in constraint_offsets:
                    self.constraint_indices[offset] = slice(None)
        self.shape = tuple(self.shape)
        # Initialize data used to implement GridStack
        self._stack_offset = 0
        self._stack_pad = 0

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

    def expand(self, values, missing=np.nan):
        """Expand an array of values to the full grid shape.
        Any values removed by a constraint will be set to NaN, by default,
        or to the specified missing value.
        """
        if values.shape != self.shape:
            raise ValueError(
                f"values shape {values.shape} does not match grid shape {self.shape}"
            )
        if self.constraint is None:
            return values
        expanded = np.full(self.expanded_shape, missing)
        # This implementation is slow since it does explicit looping
        axis0 = self.constraint_offsets[0]
        for ii in np.ndindex(self.shape):
            i0 = ii[axis0]
            jj = list(ii)
            for k in self.constraint_offsets:
                jj[k] = self.constraint_indices[k][i0]
            expanded[tuple(jj)] = values[ii]
        return expanded

    def axis(self, name):
        """Return the index of the named axis.
        Used to implement our index() method.
        """
        try:
            idx = self.names.index(name)
        except ValueError:
            raise ValueError(f'"{name}" is not in the grid')
        return idx + self._stack_offset

    def sum(self, values, keepdims=False, axis_names=None, verbose=False):
        """Sum values over our grid.
        Used for marginalization and to implement our normalize() method.
        """
        values = np.asarray(values)
        axis1 = self._stack_offset
        axis2 = axis1 + len(self.shape)
        sum_shape = values.shape[axis1:axis2]
        if sum_shape != self.shape:
            raise ValueError(
                f"values shape {values.shape} is not compatible with grid shape {self.shape}"
            )
        if axis_names is None:
            # Use all axes by default
            axes = tuple(range(axis1, axis2))
            if self.constraint is not None:
                # Multiply by constraint weights. Avoid *= so we don't modify the input.
                values = values * self.constraint_weights
        else:
            # Check for valid axis names
            try:
                axes = tuple(
                    [self.names.index(name) + self._stack_offset for name in axis_names]
                )
            except ValueError:
                raise ValueError(f"Invalid axis_names: {axis_names}")
            if self.constraint is not None:
                if values.shape != self.shape:
                    raise NotImplementedError(
                        "sum() with axis_names, constraint and GridStack not implemented"
                    )
                # Multiply by constraint weights. Avoid *= so we don't modify the input.
                values = values * self.constraint_weights
                # Expand the weighted values
                values = self.expand(values, missing=0.0)

        if verbose:
            print(f"sum: shape={values.shape} axes={axes}, keepdims={keepdims}")
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
        Used to implement our sum() method and GridStack.at()
        """
        axis = self.axis(name)
        deltas = value - self.axes[name]
        idx = np.argmin(np.abs(deltas))
        return (axis, idx)

    def getmax(self, values):
        """Return the grid coordinates of the maximum value in the specified array of values defined on this grid."""
        if values.shape != self.shape:
            raise ValueError("values must have the same shape as the grid")
        indices = list(np.unravel_index(np.argmax(values), values.shape))
        if self.constraint is not None:
            k0 = self.constraint_offsets[0]
            for k in self.constraint_offsets[1:]:
                indices[k] = indices[k0]
        return {
            name: self.axes[name].ravel()[indices[k]]
            for k, name in enumerate(self.names)
        }

    def subgrid(self, N):
        total_size = np.prod(self.shape)
        # check that N is a positive integer
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be an integer")
        num_grids = np.ceil(total_size / N)
        for i in range(int(num_grids)):
            constraint_func = lambda idx, **kwargs: (idx >= i*N) & (idx < (i+1)*N)
            subgrid = Grid(**self.axes, constraint = constraint_func, full_shape = self.shape)
            yield subgrid, subgrid.constraint_eval


def PermutationInvariant(*args):
    """Evluates constraint weights that impose permutation invariance on the grid axes passed as arguments."""
    nvars = len(args)
    arg0 = args[0].ravel()
    for i in range(1, nvars):
        if not np.array_equal(args[i].ravel(), arg0):
            raise ValueError("All axes must be identical for permutation invariance")
    size = arg0.size
    shape = np.broadcast(*args).shape
    indices = np.arange(size)
    M = np.stack(np.meshgrid(*[indices] * nvars, indexing="ij"), axis=-1).reshape(
        -1, nvars
    )
    nfact = math.factorial(nvars)

    def nperm(row):
        nrun = 1
        denom = 1
        for i in range(1, nvars):
            if row[i] < row[i - 1]:
                return 0
            elif row[i] == row[i - 1]:
                nrun += 1
            else:
                nrun = 1
            denom *= nrun
        return nfact / denom

    return np.array([nperm(row) for row in M]).reshape(shape)


def TopHat(x):
    """Helper function to define a prior that is flat within its extent."""
    x = np.asarray(x)
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be monotonically increasing")
    y = np.ones_like(x)
    y /= np.sum(y)
    return y


def CosineBump(x):
    """Convenience function to define a prior that is centrally peaked and cosine-shaped."""
    x = np.asarray(x)
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be monotonically increasing")
    xlo, xhi = np.min(x), np.max(x)
    t = 2 * np.pi * ((x - xlo) / (xhi - xlo) - 0.5)
    y = 1 + np.cos(t)
    y /= np.sum(y)
    return y

def Gaussian(x, mu, sigma):
    """Helper function to define a prior that is a Gaussian centered at mu with standard deviation sigma."""
    x = np.asarray(x)
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be monotonically increasing")
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y /= np.sum(y)
    return y

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
