"""JAX implementation of experiment design APIs."""

import math

import jax
import jax.numpy as jnp

from .grid import Grid, GridStack


def _shape_prod(shape):
    return math.prod(int(s) for s in shape)

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
        if device is None or hasattr(device, "platform"):
            self.device = device
        else:
            if not isinstance(device, str):
                raise ValueError(
                    'device must be None, "cpu", "gpu", or a jax.Device instance'
                )
            requested = device.strip().lower()
            if requested not in ("cpu", "gpu"):
                raise ValueError(
                    'device must be None, "cpu", "gpu", or a jax.Device instance'
                )
            devices = [d for d in jax.devices() if d.platform == requested]
            if not devices:
                if requested == "gpu":
                    raise RuntimeError(
                        "GPU device requested but no JAX GPU backend is available."
                    )
                raise RuntimeError(
                    "CPU device requested but no JAX CPU backend is available."
                )
            self.device = devices[0]

        self.parameters._place_on_device(self.device)
        self.features._place_on_device(self.device)
        self.designs._place_on_device(self.device)

        if mem is None:
            self.design_subgrid = int(_shape_prod(self.designs.shape))
            self.num_subgrids = 1
        else:
            if mem <= 0:
                raise ValueError("Memory limit must be positive")

            frac = mem / (
                2
                * (
                    _shape_prod(self.features.shape)
                    * _shape_prod(self.designs.shape)
                    * _shape_prod(self.parameters.shape)
                    * 8
                )
                / (1 << 20)
            )
            self.design_subgrid = int(frac * _shape_prod(self.designs.shape))
            self.num_subgrids = math.ceil(
                _shape_prod(self.designs.shape) / self.design_subgrid
            )
            if self.design_subgrid == 0:
                raise ValueError(
                    "Memory limit too low,",
                    f"invalid subgrid size: {frac * _shape_prod(self.designs.shape)} < 1",
                )

        self._initialized = False
        self.EIG = self._to_device(jnp.full(self.designs.shape, jnp.nan))

    def _to_device(self, value):
        arr = jnp.asarray(value)
        if self.device is None:
            return arr
        return jax.device_put(arr, self.device)

    def _set_masked_values(self, target, mask, values):
        mask_indices = jnp.nonzero(self._to_device(mask))
        return target.at[mask_indices].set(jnp.ravel(self._to_device(values)))

    def calculateEIG(self, prior, debug=False):
        self.prior = self._to_device(prior)
        prior_norm = self.parameters.sum(self.prior)
        if not bool(jnp.allclose(prior_norm, 1.0)):
            raise ValueError("Prior probabilities must sum to 1")

        log2prior = jnp.where(self.prior > 0, jnp.log2(self.prior), 0)
        self.H0 = float(-self.parameters.sum(self.prior * log2prior))

        for i, (s, mask) in enumerate(self.designs.subgrid(self.design_subgrid)):
            if i == 0:
                self.subgrid_shape = s.shape

            with GridStack(self.features, s, self.parameters):
                sub_likelihood = self.likelihood_func(s)
                if debug:
                    assert bool(jnp.allclose(self.features.sum(sub_likelihood), 1.0))

                self._buffer = sub_likelihood * self.prior
                marginal = self.parameters.sum(self._buffer)
                if i == 0:
                    self.marginal = self._to_device(marginal)

                if debug:
                    check = self.parameters.sum(self.likelihood_func(s) * self.prior)
                    assert bool(jnp.allclose(check, marginal))

                post_norm = self.parameters.sum(self._buffer, keepdims=True)
                self._buffer = jnp.where(
                    post_norm > 0, self._buffer / post_norm, self._buffer
                )

                if debug:
                    posterior = self.parameters.normalize(
                        self.likelihood_func(s) * self.prior
                    )
                    assert bool(jnp.allclose(posterior, self._buffer))

                log2post = jnp.where(self._buffer > 0, jnp.log2(self._buffer), 0)
                self._buffer = log2post * sub_likelihood * self.prior
                self._buffer = jnp.where(
                    post_norm > 0, self._buffer / post_norm, self._buffer
                )

                IG = self.H0 + self.parameters.sum(self._buffer)
                IG = jnp.where(jnp.reshape(post_norm, IG.shape) == 0, 0, IG)
                if i == 0:
                    self.IG = self._to_device(IG)

                if debug:
                    log2posterior = jnp.where(posterior > 0, jnp.log2(posterior), 0)
                    ig_check = self.H0 + self.parameters.sum(
                        posterior * log2posterior
                    )
                    assert bool(jnp.allclose(ig_check, IG))

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
        likelihood = self._to_device(likelihood)
        return self.features.normalize(likelihood)

    def calculateMarginalEIG(self, *nuisance_params):
        if not self._initialized:
            raise RuntimeError("Must call calculateEIG before calculateMarginalEIG")

        interest_params = tuple(
            n for n in self.parameters.names if n not in nuisance_params
        )
        prior = self.parameters.sum(
            self.prior, axis_names=nuisance_params, keepdims=True
        )
        log2prior = jnp.where(prior > 0, jnp.log2(prior), 0)
        H0 = -self.parameters.sum(prior * log2prior, axis_names=interest_params)

        EIG = self._to_device(jnp.full(self.designs.shape, jnp.nan))
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
                log2post = jnp.where(post > 0, jnp.log2(post), 0)
                IG = H0 + self.parameters.sum(
                    post * log2post, axis_names=interest_params
                )

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
                        _shape_prod(self.features.shape)
                        * _shape_prod(self.designs.shape)
                        * _shape_prod(self.parameters.shape)
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
                    _shape_prod(self.features.shape)
                    * _shape_prod(self.subgrid_shape)
                    * _shape_prod(self.parameters.shape)
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

        designs = {}
        features = {}
        for name in design_and_features:
            if name in self.designs.names:
                designs[name] = design_and_features[name]
            elif name in self.features.names:
                features[name] = design_and_features[name]

        cond_designs = Grid(**designs)
        cond_features = Grid(**features)
        cond_designs._place_on_device(self.device)
        cond_features._place_on_device(self.device)

        self._buffer = self._to_device(
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
