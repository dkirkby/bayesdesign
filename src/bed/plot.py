import numpy as np
import matplotlib.pyplot as plt

from .grid import Grid


def cornerPlot(data, grid, asize=2.5, hwspace=0.25, cmap="Blues", CL=(0.683, 0.954)):
    """Generate a matrix of plots for data specified on a grid.

    Diagonal plots show the 1D marginal distribution of the data along each axis.
    Off-diagonal plots show the 2D marginal distribution of the data along each pair of axes.

    This function works with a grid that has a constraint applied, but only points that satisfy
    the constraint are plotted, with the constraint weights applied, and all other points are
    set to zero. This will distort some or all of the 1D and 2D marginal distributions.

    Parameters
    ----------
    data : ndarray
        Data values on the grid, which must have the same shape as the grid.
    grid : Grid
        Grid object specifying the axes of the data.
    asize : float, optional
        Size of each plot in inches.
    hwspace : float, optional
        Horizontal and vertical space between plots, as a fraction of asize.
    cmap : str, optional
        Name of the colormap to use for the plots.
    CL : tuple, optional
        Confidence levels to overlay on the 2D plots. Set to None to disable.

    Returns
    -------
    tuple of (figure, axes) for the generated plot
    """
    shape = grid.shape
    if data.shape != shape:
        raise ValueError(
            f"Shape of data {data.shape} does not match the grid shape {shape}"
        )

    if not isinstance(grid, Grid):
        raise ValueError("grid must be an instance of Grid")

    if CL is not None:
        if not np.allclose(grid.sum(data), 1):
            raise ValueError(
                "Data must be normalized for CL contours. Try setting CL=None."
            )
        if not np.all(np.asarray(CL) > 0):
            raise ValueError("CL values must all be positive")
        if not np.all(np.diff(CL) > 0):
            raise ValueError("CL values must be increasing")

    # Ignore axes with length 1.
    axes = {name: axis.ravel() for name, axis in grid.axes_in.items() if axis.size > 1}
    naxes = len(axes)

    # Initialize the figure.
    fsize = naxes * asize
    figure, figaxes = plt.subplots(naxes, naxes, figsize=(fsize, fsize), squeeze=False)
    cmap = plt.get_cmap(cmap)
    ec = cmap(1.0)
    fc = cmap(0.5)

    for row, rname in enumerate(axes):
        rextent = list(axes[rname][[0, -1]])
        for col, cname in enumerate(axes):
            ax = figaxes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            cextent = list(axes[cname][[0, -1]])
            ax.set(xlim=cextent)
            if row != col:
                ax.set(ylim=rextent)
                axlist = tuple(
                    [name for k, name in enumerate(axes) if k not in (row, col)]
                )
                data2d = grid.sum(data, axis_names=axlist).T
                vmax = 1.5 * data2d.max()
                ax.imshow(
                    data2d,
                    origin="lower",
                    interpolation="none",
                    vmin=0,
                    vmax=vmax,
                    extent=cextent + rextent,
                    aspect="auto",
                    cmap=cmap,
                )
                if CL is not None:
                    # Sort 2D bin values.
                    values = np.sort(data2d, axis=None)[::-1]
                    sums = np.cumsum(values)
                    # Interpolate to get CL contour levels.
                    levels = np.interp(CL, sums, values)[::-1]
                    # Overlay contour levels.
                    ax.contour(
                        axes[cname],
                        axes[rname],
                        data2d,
                        levels=levels,
                        colors=(ec,),
                        linestyles=("-", "--", ":")[: len(levels)][::-1],
                    )
            else:
                axlist = tuple([name for k, name in enumerate(axes) if k != row])
                data1d = grid.sum(data, axis_names=axlist)
                ax.fill_between(axes[rname], data1d, ec="none", fc=fc)
                ax.plot(axes[rname], data1d, c=ec)
                ax.set(ylim=(0, None), yticks=[])
            if col == 0 and row > 0:
                ax.set(ylabel=rname)
            if row == naxes - 1:
                ax.set(xlabel=cname)
    plt.subplots_adjust(right=0.99, top=0.99, wspace=hwspace, hspace=hwspace)

    return figure, figaxes
