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

__version__ = "0.7.0"

__all__ = [
    "Grid",
    "GridStack",
    "PermutationInvariant",
    "TopHat",
    "CosineBump",
    "Gaussian",
    "ExperimentDesigner",
]
