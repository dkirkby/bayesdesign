"""JAX implementation of the bayesdesign public API."""

from .design import ExperimentDesigner
from .grid import (
    CosineBump,
    Gaussian,
    Grid,
    GridStack,
    PermutationInvariant,
    TopHat,
)

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
