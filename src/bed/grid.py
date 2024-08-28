"""Utility functions for working with discrete grids of variables."""

import numpy as np


class Grid:

    def __init__(self, **axes):
        self.axes = {}
        mesh = []
        self.names = []
        self.offsets = []
        self.keep_groups = []
        for name in axes:
            # check for a pseudo-name of the form keep__<name1>__<name2>__...
            tokens = name.split("__")
            if len(tokens) >= 3 and tokens[0] == "keep":
                group = {}
                for tname in tokens[1:]:
                    try:
                        idx = self.names.index(tname)
                    except ValueError:
                        raise ValueError(f'Unknown name "{tname}" in {name}')
                    group[tname] = self.axes[tname]
                # Build a dictionary with keys name1, name2, ... and corresponding values for a fully populated mesh grid.
                full_group = dict(
                    zip(
                        group.keys(),
                        np.stack(
                            [
                                G
                                for G in np.meshgrid(
                                    *group.values(),
                                    sparse=False,
                                    copy=True,
                                    indexing="ij",
                                )
                            ]
                        ).reshape(len(group), -1),
                    )
                )
                # Use the provided function to filter each array to just those combinations we should keep.
                try:
                    keep = axes[name](**full_group)
                    keep_group = {
                        tname: axis[keep] for tname, axis in full_group.items()
                    }
                except Exception as e:
                    raise RuntimeError(f"{name} filter failed: {e}")
                self.keep_groups.append(keep_group)
                # Replace original unfiltered axes in this group.
                offset = None
                for tname, axis in keep_group.items():
                    # Replace original axis definition
                    self.axes[tname] = axis
                    idx = self.names.index(tname)
                    if offset is None:
                        offset = self.offsets[idx]
                    else:
                        self.offsets[idx] = offset
                continue
            # treat this like a new named axis
            axis = np.atleast_1d(axes[name])
            if axis.ndim != 1:
                raise RuntimeError(f'Invalid grid for axis "{name}"')
            self.offsets.append(len(self.axes))
            self.axes[name] = axis
            self.names.append(name)
        # Offset each axis for broadcasting
        naxes = len(self.names)
        bcast = 1
        for i, (name, axis) in enumerate(self.axes.items()):
            self.axes[name] = axis.reshape([-1] + [1] * (naxes - self.offsets[i] - 1))
            bcast = bcast * self.axes[name]
        self.shape = bcast.shape
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

    def axis(self, name):
        """Return the index of the named axis. Axes in the same keep group will have the same index.
        Used to implement our index() method.
        """
        try:
            idx = self.names.index(name)
        except ValueError:
            raise ValueError(f'"{name}" is not in the grid')
        return self.offsets[idx] + self._stack_offset

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
