"""Utility functions for working with discrete grids of variables."""

import inspect
import itertools
import math

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError(
        "bed_jax requires JAX. Install with `pip install -e '.[jax-cpu]'` "
        "or `pip install -e '.[jax]'."
    ) from exc

from .util import resolve_device


class Grid:

    def __init__(self, constraint=None, full_shape=None, device=None, **axes):
        self.device = resolve_device(device)
        self.axes = {}
        self.axes_in = {}
        naxes = len(axes)
        for offset, name in enumerate(axes):
            # Check for a valid axis definition.
            with jax.default_device(self.device):
                axis = jnp.atleast_1d(jnp.asarray(axes[name]))
            # Store the axis offset to its position in the grid.
            shape = [1] * offset + [-1] + [1] * (naxes - offset - 1)
            self.axes[name] = axis.reshape(shape)
            # Remember the original axis definition.
            self.axes_in[name] = axis

        self.names = tuple(self.axes.keys())
        self.shape = [int(axis.size) for axis in self.axes.values()]
        self.expanded_shape = tuple(self.shape)
        self.constraint = constraint

        # Check for a constraint function.
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

            # Evaluate the constraint function on the full grid.
            self.constraint_args = {
                name: self.axes[name] for name in constraint_names if name in self.axes
            }
            if "idx" in constraint_names:
                if full_shape is None:
                    raise ValueError(
                        "full_shape is required when constraint uses idx"
                    )
                with jax.default_device(self.device):
                    idx = jnp.arange(math.prod(full_shape)).reshape(full_shape)
                self.constraint_args["idx"] = idx

            with jax.default_device(self.device):
                constraint_eval = jnp.asarray(constraint(**self.constraint_args))
            self.constraint_eval = constraint_eval
            if bool(jnp.any(constraint_eval < 0)):
                raise ValueError("constraint must be non-negative")

            # Replace constrained axes with the reduced set of values.
            if "idx" in constraint_names:
                squeezed = jnp.squeeze(constraint_eval)
                nonzero = jnp.nonzero(squeezed)
                if len(nonzero) == len(self.names):
                    mapper = {
                        name: nonzero[offset]
                        for offset, name in enumerate(self.names)
                    }
                else:
                    # Keep a robust fallback for edge cases where squeeze()
                    # changes dimensionality in ways that do not match names.
                    flat_nonzero = jnp.nonzero(jnp.ravel(squeezed))[0]
                    mapper = {name: flat_nonzero for name in self.names}
            else:
                nonzero = jnp.nonzero(jnp.squeeze(constraint_eval))
                mapper = {
                    name: indices for name, indices in zip(constraint_names, nonzero)
                }

            constraint_offsets = []
            reduced_shape = None
            for offset, name in enumerate(self.names):
                if name not in constraint_names:
                    continue
                if not constraint_offsets:
                    reduced_shape = [1] * offset + [-1] + [1] * (naxes - offset - 1)
                    self.shape[offset] = int(jnp.count_nonzero(constraint_eval))
                else:
                    self.shape[offset] = 1
                self.axes[name] = jnp.ravel(self.axes[name])[mapper[name]].reshape(
                    reduced_shape
                )
                constraint_offsets.append(offset)

            first_offset = constraint_offsets[0]
            weights_shape = tuple(
                [1] * first_offset + [-1] + [1] * (naxes - first_offset - 1)
            )
            mask = constraint_eval != 0
            self.constraint_weights = jnp.ravel(constraint_eval[mask]).reshape(
                weights_shape
            )

            # Compute and save indices needed to expand arrays tabulated on
            # the reduced grid.
            self.constraint_offsets = tuple(constraint_offsets)
            self.constraint_indices = list(jnp.nonzero(constraint_eval))
            for offset in range(naxes):
                if offset not in constraint_offsets:
                    self.constraint_indices[offset] = slice(None)

        self.shape = tuple(self.shape)
        # Initialize data used to implement GridStack.
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
                    for (_, (name, axis)) in enumerate(self.axes.items())
                ]
            )
            + "]"
        )

    def __getattr__(self, name):
        """Return a 1D array of values for the named axis."""
        axis = self.axes[name]
        return axis.reshape(axis.shape + tuple([1] * self._stack_pad))

    def expand(self, values, missing=jnp.nan):
        """Expand an array of values to the full grid shape.

        Any values removed by a constraint will be set to NaN, by default,
        or to the specified missing value.
        """
        if tuple(values.shape) != self.shape:
            raise ValueError(
                f"values shape {values.shape} does not match grid shape {self.shape}"
            )

        values = jnp.asarray(values)
        if self.constraint is None:
            return values

        fill_dtype = jnp.result_type(values.dtype, jnp.asarray(missing).dtype)
        with jax.default_device(self.device):
            expanded = jnp.full(self.expanded_shape, missing, dtype=fill_dtype)

        # This implementation is slow since it does explicit looping.
        axis0 = self.constraint_offsets[0]
        for ii in itertools.product(*(range(dim) for dim in self.shape)):
            i0 = ii[axis0]
            jj = list(ii)
            for k in self.constraint_offsets:
                jj[k] = int(self.constraint_indices[k][i0])
            expanded = expanded.at[tuple(jj)].set(values[ii])

        return expanded

    def axis(self, name):
        """Return the index of the named axis.

        Used to implement our index() method.
        """
        try:
            idx = self.names.index(name)
        except ValueError as exc:
            raise ValueError(f'"{name}" is not in the grid') from exc
        return idx + self._stack_offset

    def sum(self, values, keepdims=False, axis_names=None, verbose=False):
        """Sum values over our grid.

        Used for marginalization and to implement our normalize() method.
        """
        with jax.default_device(self.device):
            values = jnp.asarray(values)
            axis1 = self._stack_offset
            axis2 = axis1 + len(self.shape)
            sum_shape = values.shape[axis1:axis2]

            if axis_names is None:
                # Use all axes by default.
                if sum_shape != self.shape:
                    raise ValueError(
                        f"values shape {values.shape} is not compatible with grid shape {self.shape}"
                    )
                axes = tuple(range(axis1, axis2))
                if self.constraint is not None:
                    # Multiply by constraint weights. Avoid *= so we do not
                    # modify the input array.
                    values = values * jnp.asarray(self.constraint_weights)
            else:
                # Check for valid axis names and named-axis compatible sizes.
                try:
                    axes = tuple(
                        [self.names.index(name) + self._stack_offset for name in axis_names]
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid axis_names: {axis_names}") from exc

                if len(sum_shape) != len(self.shape) or any(
                    sum_shape[self.names.index(name)]
                    != self.shape[self.names.index(name)]
                    for name in axis_names
                ):
                    raise ValueError(
                        f"values shape {values.shape} is not compatible with grid shape {self.shape}"
                    )

                if self.constraint is not None:
                    if tuple(values.shape) != self.shape:
                        raise NotImplementedError(
                            "sum() with axis_names, constraint and GridStack not implemented"
                        )
                    # Multiply by constraint weights. Avoid *= so we do not
                    # modify the input array.
                    values = values * jnp.asarray(self.constraint_weights)
                    # Expand the weighted values.
                    values = self.expand(values, missing=0.0)

            if verbose:
                print(f"sum: shape={values.shape} axes={axes}, keepdims={keepdims}")

            if sum_shape != self.shape:
                # Summing specific axis_names: use keepdims=True internally,
                # then squeeze axes that are size 1 but not naturally
                # single-value axes.
                result = jnp.sum(values, axis=axes, keepdims=True)
                if not keepdims:
                    squeeze_axes = tuple(
                        axis1 + i
                        for i in range(len(self.shape))
                        if result.shape[axis1 + i] == 1 and self.shape[i] != 1
                    )
                    if squeeze_axes:
                        result = jnp.squeeze(result, axis=squeeze_axes)
                return result

            return jnp.sum(values, axis=axes, keepdims=keepdims)

    def normalize(self, values):
        """Normalize the array values over our grid."""
        with jax.default_device(self.device):
            values = jnp.asarray(values)
            norm = self.sum(values, keepdims=True)
            return values / norm

    def extent(self, name):
        """Return the (min,max) extent of the named axis.

        Useful to set plot axis limits.
        """
        axis = self.axes[name]
        return (float(jnp.min(axis)), float(jnp.max(axis)))

    def index(self, name, value):
        """Return the index of the named axis and closest location along that axis."""
        axis = self.axis(name)
        deltas = value - self.axes[name]
        idx = int(jnp.argmin(jnp.abs(deltas)))
        return (axis, idx)

    def getmax(self, values):
        """Return the grid coordinates of the maximum value in values."""
        if tuple(values.shape) != self.shape:
            raise ValueError("values must have the same shape as the grid")

        flat_index = int(jnp.argmax(jnp.asarray(values)))
        indices = [int(i) for i in jnp.unravel_index(flat_index, values.shape)]
        if self.constraint is not None:
            k0 = self.constraint_offsets[0]
            for k in self.constraint_offsets[1:]:
                indices[k] = indices[k0]

        out = {}
        for k, name in enumerate(self.names):
            value = jnp.ravel(self.axes[name])[indices[k]]
            out[name] = value.item()
        return out

    def subgrid(self, N):
        total_size = math.prod(self.shape)
        # Check that N is a positive integer.
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be an integer")

        num_grids = math.ceil(total_size / N)
        for i in range(num_grids):
            constraint_func = (
                lambda idx, _i=i, _N=N, **kwargs: (idx >= _i * _N)
                & (idx < (_i + 1) * _N)
            )
            subgrid = Grid(
                **self.axes,
                constraint=constraint_func,
                full_shape=self.shape,
                device=self.device,
            )
            yield subgrid, subgrid.constraint_eval


def PermutationInvariant(*args):
    """Evaluate constraint weights that impose permutation invariance."""
    nvars = len(args)
    args = [jnp.asarray(arg) for arg in args]
    arg0 = jnp.ravel(args[0])
    for i in range(1, nvars):
        if not bool(jnp.array_equal(jnp.ravel(args[i]), arg0)):
            raise ValueError("All axes must be identical for permutation invariance")

    size = int(arg0.size)
    shape = jnp.broadcast_arrays(*args)[0].shape
    indices = jnp.arange(size)
    M = jnp.stack(jnp.meshgrid(*[indices] * nvars, indexing="ij"), axis=-1).reshape(
        -1, nvars
    )
    nfact = math.factorial(nvars)

    def nperm(row):
        row = [int(v) for v in row.tolist()]
        nrun = 1
        denom = 1
        for i in range(1, nvars):
            if row[i] < row[i - 1]:
                return 0.0
            if row[i] == row[i - 1]:
                nrun += 1
            else:
                nrun = 1
            denom *= nrun
        return nfact / denom

    out = jnp.array([nperm(row) for row in M], dtype=jnp.float64).reshape(shape)
    return out


def TopHat(x):
    """Helper function to define a prior that is flat within its extent."""
    x = jnp.asarray(x)
    if not bool(jnp.all(jnp.diff(x) > 0)):
        raise ValueError("x must be monotonically increasing")
    with jax.default_device(x.device):
        y = jnp.ones_like(x)
        y = y / jnp.sum(y)
    return y


def CosineBump(x):
    """Convenience function for a centrally peaked cosine-shaped prior."""
    x = jnp.asarray(x)
    if not bool(jnp.all(jnp.diff(x) > 0)):
        raise ValueError("x must be monotonically increasing")
    xlo, xhi = jnp.min(x), jnp.max(x)
    t = 2 * jnp.pi * ((x - xlo) / (xhi - xlo) - 0.5)
    y = 1 + jnp.cos(t)
    y = y / jnp.sum(y)
    return y


def Gaussian(x, mu, sigma):
    """Helper function to define a Gaussian prior centered at mu."""
    x = jnp.asarray(x)
    if not bool(jnp.all(jnp.diff(x) > 0)):
        raise ValueError("x must be monotonically increasing")
    y = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y = y / jnp.sum(y)
    return y


class GridStack:
    """Context manager to temporarily stack grids together."""

    def __init__(self, *grids):
        self.grids = grids

    def __str__(self):
        return "[" + ",".join([str(grid) for grid in self.grids]) + "]"

    def __repr__(self):
        return "[" + ", ".join([repr(grid) for grid in self.grids]) + "]"

    def __enter__(self):
        offset = sum([len(grid.axes) for grid in self.grids])
        pad = 0
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
