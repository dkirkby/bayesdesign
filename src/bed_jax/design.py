"""JAX implementation of experiment design APIs."""

import math

import jax
import jax.numpy as jnp

from .grid import Grid, GridStack
from .util import resolve_device


class ExperimentDesigner:
    """Brute-force expected information gain calculation using JAX arrays."""

    def __init__(
        self,
        parameters,
        features,
        designs,
        unnorm_lfunc,
        lfunc_args={},
        mem=None,
        device=None,
    ):
        self.parameters = parameters
        self.features = features
        self.designs = designs
        self.unnorm_lfunc = unnorm_lfunc
        self.lfunc_args = lfunc_args
        self.device = resolve_device(device)

        for name, grid in (
            ("parameters", self.parameters),
            ("features", self.features),
            ("designs", self.designs),
        ):
            if grid.device != self.device:
                raise ValueError(
                    f"Grid {name} device ({grid.device.platform}) does not match ExperimentDesigner device ({self.device.platform})."
                )

        if mem is None:
            self.design_subgrid = int(math.prod(self.designs.shape))
            self.num_subgrids = 1
        else:
            if mem <= 0:
                raise ValueError("Memory limit must be positive")

            # Calculate the fractional decrease required to meet the memory
            # limit, accounting for the buffer doubling memory usage.
            frac = mem / (
                2
                * (
                    math.prod(self.features.shape)
                    * math.prod(self.designs.shape)
                    * math.prod(self.parameters.shape)
                    * 8
                )
                / (1 << 20)
            )
            self.design_subgrid = int(frac * math.prod(self.designs.shape))
            self.num_subgrids = math.ceil(
                math.prod(self.designs.shape) / self.design_subgrid
            )
            if self.design_subgrid == 0:
                raise ValueError(
                    "Memory limit too low,",
                    f"invalid subgrid size: {frac * math.prod(self.designs.shape)} < 1",
                )

        self._initialized = False
        with jax.default_device(self.device):
            self.EIG = jnp.full(self.designs.shape, jnp.nan)

    def _set_masked_values(self, target, mask, values):
        with jax.default_device(self.device):
            mask_indices = jnp.nonzero(jnp.asarray(mask))
            return target.at[mask_indices].set(jnp.ravel(jnp.asarray(values)))

    def calculateEIG(self, prior, debug=False):
        with jax.default_device(self.device):
            self.prior = jnp.asarray(prior)
        prior_norm = self.parameters.sum(self.prior)
        if not bool(jnp.allclose(prior_norm, 1.0)):
            raise ValueError("Prior probabilities must sum to 1")

        # Calculate prior entropy in bits (careful with x log x = 0 for x=0).
        log2prior = jnp.where(self.prior > 0, jnp.log2(self.prior), 0)
        self.H0 = float(-self.parameters.sum(self.prior * log2prior))

        for i, (s, mask) in enumerate(self.designs.subgrid(self.design_subgrid)):
            if i == 0:
                # Store first subgrid likelihood shape for describe().
                self.subgrid_shape = s.shape

            with GridStack(self.features, s, self.parameters):
                sub_likelihood = self.likelihood_func(s)  # use of memory
                assert bool(jnp.allclose(self.features.sum(sub_likelihood), 1.0))

                # Tabulate the marginal probability P(y|xi) by integrating P(y|theta,xi) P(theta) over theta.
                # No explicit normalization is required since P(y|theta,xi) and P(theta) are already normalized.
                # Check that the likelihood is normalized over features
                self._buffer = sub_likelihood * self.prior
                marginal = self.parameters.sum(self._buffer)
                if i == 0:
                    # Store first subgrid marginal for describe().
                    self.marginal = marginal

                if debug:
                    assert bool(
                        jnp.allclose(
                            self.parameters.sum(
                                self.likelihood_func(s) * self.prior
                            ),
                            marginal,
                        )
                    ), "marginal check failed"

                # Tabulate the posterior P(theta|y,xi) by normalizing P(y|theta,xi) P(theta) over parameters.
                # Use the prior for any (design, feature) points where the likelihood x prior is zero
                # so the corresponding information gain is zero.
                post_norm = self.parameters.sum(self._buffer, keepdims=True)
                self._buffer = jnp.where(
                    post_norm > 0, self._buffer / post_norm, self._buffer
                )

                if debug:
                    # This will fail if likelihood*prior is zero for any
                    # (design, feature) point.
                    posterior = self.parameters.normalize(
                        self.likelihood_func(s) * self.prior
                    )
                    assert bool(
                        jnp.allclose(posterior, self._buffer)
                    ), "posterior check failed"

                # Tabulate information gain in bits IG = post * log2(post) + H0.
                # Do calculations in stages to avoid large temporaries.
                log2post = jnp.where(self._buffer > 0, jnp.log2(self._buffer), 0)
                self._buffer = log2post * sub_likelihood * self.prior
                self._buffer = jnp.where(
                    post_norm > 0, self._buffer / post_norm, self._buffer
                )

                IG = self.H0 + self.parameters.sum(self._buffer)
                IG = jnp.where(jnp.reshape(post_norm, IG.shape) == 0, 0, IG)
                if i == 0:
                    self.IG = IG

                if debug:
                    log2posterior = jnp.where(posterior > 0, jnp.log2(posterior), 0)
                    assert bool(
                        jnp.allclose(
                            self.H0
                            + self.parameters.sum(posterior * log2posterior),
                            IG,
                        )
                    ), "IG check failed"

                eig = self.features.sum(marginal * IG)
                self.EIG = self._set_masked_values(self.EIG, mask, eig)

        self._initialized = True
        if hasattr(self, "_buffer"):
            del self._buffer
        return self.designs.getmax(self.EIG)

    def likelihood_func(self, s):
        likelihood = self.unnorm_lfunc(
            self.parameters, self.features, s, **self.lfunc_args
        )
        with jax.default_device(self.device):
            likelihood = jnp.asarray(likelihood)
        return self.features.normalize(likelihood)

    def calculateMarginalEIG(self, *nuisance_params):
        if not self._initialized:
            raise RuntimeError("Must call calculateEIG before calculateMarginalEIG")

        # Determine parameters of interest (complement of nuisance params).
        interest_params = tuple(
            n for n in self.parameters.names if n not in nuisance_params
        )
        # Calculate marginal prior and its entropy.
        prior = self.parameters.sum(
            self.prior, axis_names=nuisance_params, keepdims=True
        )
        log2prior = jnp.where(prior > 0, jnp.log2(prior), 0)
        H0 = -self.parameters.sum(prior * log2prior, axis_names=interest_params)

        # Calculate marginal posterior and information gain.
        with jax.default_device(self.device):
            EIG = jnp.full_like(self.EIG, jnp.nan)
        for (s, mask) in self.designs.subgrid(self.design_subgrid):
            with GridStack(self.features, s, self.parameters):
                sub_likelihood = self.likelihood_func(s)
                self._buffer = sub_likelihood * self.prior
                marginal = self.parameters.sum(self._buffer)
                post_norm = self.parameters.sum(self._buffer, keepdims=True)

                self._buffer = jnp.where(
                    post_norm > 0, self._buffer / post_norm, self._buffer
                )
                self._buffer = jnp.where(post_norm == 0, self.prior, self._buffer)

                post = self.parameters.sum(
                    self._buffer, axis_names=nuisance_params, keepdims=True
                )
                # Calculate IG for all possible designs and measurements.
                log2post = jnp.where(post > 0, jnp.log2(post), 0)
                IG = H0 + self.parameters.sum(
                    post * log2post, axis_names=interest_params
                )

                # Tabulate expected information gain in bits.
                eig = self.features.sum(marginal * IG)
                EIG = self._set_masked_values(EIG, mask, eig)

        if hasattr(self, "_buffer"):
            del self._buffer
        return EIG

    def describe(self):
        if not self._initialized:
            print("Not initialized")
            return

        for name in ("designs", "features", "parameters"):
            grid = self.__dict__[name]
            if name == "designs" and self.num_subgrids > 1:
                print(
                    f"GRID  {name:>16s} {repr(grid)}, {self.num_subgrids} subgrids used"
                )
            else:
                print(f"GRID  {name:>16s} {repr(grid)}")

        for name in ("prior", "subgrid_shape", "marginal", "IG", "EIG"):
            array = self.__dict__[name]
            if name == "subgrid_shape":
                if self.num_subgrids > 1:
                    full_likelihood_name = "full_likelihood"
                    full_likelihood_shape = str(
                        self.features.shape
                        + self.designs.shape
                        + self.parameters.shape
                    )
                    full_likelihood_size = (
                        math.prod(self.features.shape)
                        * math.prod(self.designs.shape)
                        * math.prod(self.parameters.shape)
                        * 8
                        / (1 << 20)
                    )
                    print(
                        f"ARRAY {full_likelihood_name:>16s} {full_likelihood_shape:26s} {full_likelihood_size:9.1f} Mb"
                    )
                name = "likelihood"
                likelihood_shape = str(
                    self.features.shape
                    + self.subgrid_shape
                    + self.parameters.shape
                )
                likelihood_size = (
                    math.prod(self.features.shape)
                    * math.prod(self.subgrid_shape)
                    * math.prod(self.parameters.shape)
                    * 8
                    / (1 << 20)
                )
                print(
                    f"ARRAY {name:>16s} {likelihood_shape:26s} {likelihood_size:9.1f} Mb"
                )
            else:
                nbytes = jnp.asarray(array).nbytes
                print(
                    f"ARRAY {name:>16s} {repr(array.shape):26s} {nbytes/(1<<20):9.1f} Mb"
                )

    def get_posterior(self, **design_and_features):
        if not self._initialized:
            print("Not initialized")
            return

        # Match input names to design/feature grids.
        designs = {}
        features = {}
        for name in design_and_features:
            if name in self.designs.names:
                designs[name] = design_and_features[name]
            elif name in self.features.names:
                features[name] = design_and_features[name]

        cond_designs = Grid(device=self.device, **designs)
        cond_features = Grid(device=self.device, **features)

        with jax.default_device(self.device):
            self._buffer = jnp.asarray(
                self.unnorm_lfunc(
                    self.parameters,
                    cond_features,
                    cond_designs,
                    **self.lfunc_args,
                )
            )
        self._buffer = self._buffer * self.prior
        post_norm = self.parameters.sum(self._buffer, keepdims=True)
        self._buffer = jnp.where(
            post_norm > 0, self._buffer / post_norm, self._buffer
        )
        self._buffer = jnp.where(post_norm == 0, self.prior, self._buffer)
        return self._buffer

    def update(self, **design_and_features):
        post = self.get_posterior(**design_and_features)
        return self.calculateEIG(post)
