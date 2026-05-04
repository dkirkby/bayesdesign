"""JAX implementation of the bayesdesign public API."""

__version__ = "0.6.0dev"

__all__ = [
    "Grid",
    "GridStack",
    "PermutationInvariant",
    "TopHat",
    "CosineBump",
    "Gaussian",
    "ExperimentDesigner",
]

_design_names = frozenset({"ExperimentDesigner"})
_grid_names = frozenset(
    {"Grid", "GridStack", "PermutationInvariant", "TopHat", "CosineBump", "Gaussian"}
)


def __getattr__(name):
    if name in _design_names:
        from .design import ExperimentDesigner

        globals()["ExperimentDesigner"] = ExperimentDesigner
        return ExperimentDesigner
    if name in _grid_names:
        from .grid import (
            CosineBump,
            Gaussian,
            Grid,
            GridStack,
            PermutationInvariant,
            TopHat,
        )

        _imported = {
            "CosineBump": CosineBump,
            "Gaussian": Gaussian,
            "Grid": Grid,
            "GridStack": GridStack,
            "PermutationInvariant": PermutationInvariant,
            "TopHat": TopHat,
        }
        globals().update(_imported)
        return _imported[name]
    raise AttributeError(f"module 'bed' has no attribute {name!r}")
