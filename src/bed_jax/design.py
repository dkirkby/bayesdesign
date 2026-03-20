"""JAX implementation of experiment design APIs."""

import math

import jax
import jax.numpy as jnp

from .grid import Grid, GridStack
from .util import resolve_device


class _AxisBundle:
    """Minimal attribute-access wrapper for array-valued grid axes."""

    __slots__ = ("_axes",)

    def __init__(self, axes):
        self._axes = axes

    def __getattr__(self, name):
        try:
            return self._axes[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


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
        self._compiled_kernels = {
            "calculateEIG": {},
            "calculateEIG_chunked": {},
            "calculateMarginalEIG": {},
            "calculateMarginalEIG_chunked": {},
            "get_posterior": {},
        }
        with jax.default_device(self.device):
            self.EIG = jnp.full(self.designs.shape, jnp.nan)

    def _grid_weights(self, grid):
        with jax.default_device(self.device):
            if grid.constraint is None:
                return jnp.ones(grid.shape)
            return jnp.broadcast_to(jnp.asarray(grid.constraint_weights), grid.shape)

    def _pad_subgrid_values(self, values, actual_shape, target_shape):
        if tuple(actual_shape) == tuple(target_shape):
            return values

        ndim = len(self.features.shape) + len(actual_shape) + len(self.parameters.shape)
        pad_width = [(0, 0)] * ndim
        design_offset = len(self.features.shape)
        for axis, (actual, target) in enumerate(zip(actual_shape, target_shape)):
            pad_width[design_offset + axis] = (0, int(target) - int(actual))
        return jnp.pad(values, pad_width)

    def _broadcast_grid_axes(self, grid):
        return {
            name: jnp.broadcast_to(jnp.asarray(grid.axes[name]), grid.shape)
            for name in grid.names
        }

    def _stacked_axis_bundle(self, grid, total_ndim, offset):
        axes = {}
        for name, values in self._broadcast_grid_axes(grid).items():
            shape = [1] * total_ndim
            shape[offset : offset + len(grid.shape)] = grid.shape
            axes[name] = jnp.reshape(values, tuple(shape))
        return _AxisBundle(axes)

    def _flattened_design_axes(self):
        return {
            name: jnp.reshape(values, (-1,))
            for name, values in self._broadcast_grid_axes(self.designs).items()
        }

    def _get_chunk_scan_setup(self):
        key = int(self.design_subgrid)
        cache = self._compiled_kernels.setdefault("chunk_scan_setup", {})
        if key in cache:
            return cache[key]

        feature_ndim = len(self.features.shape)
        param_ndim = len(self.parameters.shape)
        total_ndim = feature_ndim + 1 + param_ndim
        design_offset = feature_ndim
        chunk_shape = (key,)
        total_designs = int(math.prod(self.designs.shape))
        num_chunks = math.ceil(total_designs / key)
        padded_designs = num_chunks * key
        expected_shape = self.features.shape + chunk_shape + self.parameters.shape
        feature_axes = tuple(range(feature_ndim))
        feature_weights = jnp.reshape(
            self._grid_weights(self.features),
            self.features.shape + (1,) * (1 + param_ndim),
        )
        params_bundle = self._stacked_axis_bundle(
            self.parameters, total_ndim, feature_ndim + 1
        )
        features_bundle = self._stacked_axis_bundle(self.features, total_ndim, 0)
        flat_design_axes = self._flattened_design_axes()
        padded_design_axes = {
            name: jnp.pad(values, (0, padded_designs - total_designs))
            for name, values in flat_design_axes.items()
        }

        cache[key] = {
            "chunk_shape": chunk_shape,
            "total_ndim": total_ndim,
            "design_offset": design_offset,
            "total_designs": total_designs,
            "num_chunks": num_chunks,
            "expected_shape": expected_shape,
            "feature_axes": feature_axes,
            "feature_weights": feature_weights,
            "params_bundle": params_bundle,
            "features_bundle": features_bundle,
            "padded_design_axes": padded_design_axes,
        }
        return cache[key]

    def _get_marginal_grouping(self, nuisance_axes):
        key = tuple(int(axis) for axis in nuisance_axes)
        cache = self._compiled_kernels.setdefault("marginal_groups", {})
        if key in cache:
            return cache[key]

        param_shape = tuple(int(v) for v in self.parameters.shape)
        full_shape = tuple(int(v) for v in self.parameters.expanded_shape)
        ndim = len(param_shape)
        interest_axes = tuple(axis for axis in range(ndim) if axis not in nuisance_axes)
        param_size = int(math.prod(param_shape))

        if self.parameters.constraint is None:
            reduced_indices = jnp.unravel_index(jnp.arange(param_size), param_shape)
            full_indices = tuple(
                jnp.asarray(idx, dtype=jnp.int32) for idx in reduced_indices
            )
        else:
            weights = jnp.asarray(self.parameters.constraint_weights)
            if not bool(jnp.allclose(weights, 1.0)):
                raise NotImplementedError(
                    "calculateMarginalEIG does not support weighted constrained parameter grids."
                )

            reduced_indices = jnp.unravel_index(jnp.arange(param_size), param_shape)
            primary_axis = self.parameters.constraint_offsets[0]
            full_indices = []
            for axis in range(ndim):
                if axis in self.parameters.constraint_offsets:
                    full_idx = jnp.asarray(
                        self.parameters.constraint_indices[axis],
                        dtype=jnp.int32,
                    )[reduced_indices[primary_axis]]
                else:
                    full_idx = jnp.asarray(reduced_indices[axis], dtype=jnp.int32)
                full_indices.append(full_idx)
            full_indices = tuple(full_indices)

        if interest_axes:
            interest_shape = tuple(full_shape[axis] for axis in interest_axes)
            group_ids = jnp.ravel_multi_index(
                tuple(full_indices[axis] for axis in interest_axes), interest_shape
            ).astype(jnp.int32)
            num_segments = int(math.prod(interest_shape))
        else:
            interest_shape = ()
            group_ids = jnp.zeros((param_size,), dtype=jnp.int32)
            num_segments = 1

        cache[key] = (
            tuple(int(axis) for axis in interest_axes),
            group_ids,
            num_segments,
            interest_shape,
        )
        return cache[key]

    def _make_eig_eval(self, subgrid_shape):
        key = tuple(int(v) for v in subgrid_shape)
        feature_shape = tuple(int(v) for v in self.features.shape)
        param_shape = tuple(int(v) for v in self.parameters.shape)
        feature_size = int(math.prod(feature_shape))
        subgrid_size = int(math.prod(subgrid_shape))
        param_weights = self._grid_weights(self.parameters)
        feature_weights = jnp.reshape(self._grid_weights(self.features), (feature_size,))

        def kernel(sub_likelihood, prior, h0):
            flat = jnp.reshape(
                sub_likelihood,
                (feature_size, subgrid_size) + param_shape,
            )
            design_first = jnp.swapaxes(flat, 0, 1)

            def one_feature(likelihood_row):
                weighted = likelihood_row * prior
                marginal = jnp.sum(weighted * param_weights)
                posterior = jnp.where(marginal > 0, weighted / marginal, weighted)
                log2post = jnp.where(posterior > 0, jnp.log2(posterior), 0.0)
                ig = h0 + jnp.sum(posterior * log2post * param_weights)
                ig = jnp.where(marginal > 0, ig, 0.0)
                return marginal, ig

            def one_design(feature_rows):
                marginals, ig_values = jax.vmap(one_feature)(feature_rows)
                eig = jnp.sum(feature_weights * marginals * ig_values)
                return marginals, ig_values, eig

            marginals, ig_values, eig_values = jax.vmap(one_design)(design_first)
            marginal = jnp.swapaxes(marginals, 0, 1).reshape(feature_shape + key)
            ig = jnp.swapaxes(ig_values, 0, 1).reshape(feature_shape + key)
            eig = jnp.reshape(eig_values, key)
            return marginal, ig, eig

        return kernel

    def _get_eig_kernel(self, subgrid_shape):
        key = tuple(int(v) for v in subgrid_shape)
        cache = self._compiled_kernels["calculateEIG"]
        if key in cache:
            return cache[key]

        cache[key] = jax.jit(self._make_eig_eval(key))
        return cache[key]

    def _get_chunked_eig_kernel(self):
        key = int(self.design_subgrid)
        cache = self._compiled_kernels["calculateEIG_chunked"]
        if key in cache:
            return cache[key]

        setup = self._get_chunk_scan_setup()
        chunk_shape = setup["chunk_shape"]
        total_ndim = setup["total_ndim"]
        design_offset = setup["design_offset"]
        total_designs = setup["total_designs"]
        num_chunks = setup["num_chunks"]
        expected_shape = setup["expected_shape"]
        feature_axes = setup["feature_axes"]
        feature_weights = setup["feature_weights"]
        params_bundle = setup["params_bundle"]
        features_bundle = setup["features_bundle"]
        padded_design_axes = setup["padded_design_axes"]
        eval_chunk = self._make_eig_eval(chunk_shape)

        def kernel(prior, h0):
            chunk_indices = jnp.arange(num_chunks, dtype=jnp.int32)

            def one_chunk(_, chunk_idx):
                start = chunk_idx * key
                design_axes = {}
                for name, values in padded_design_axes.items():
                    chunk_values = jax.lax.dynamic_slice_in_dim(values, start, key)
                    shape = [1] * total_ndim
                    shape[design_offset] = key
                    design_axes[name] = jnp.reshape(chunk_values, tuple(shape))
                designs_bundle = _AxisBundle(design_axes)

                likelihood = self.unnorm_lfunc(
                    params_bundle,
                    features_bundle,
                    designs_bundle,
                    **self.lfunc_args,
                )
                likelihood = jnp.asarray(likelihood)
                if tuple(likelihood.shape) != expected_shape:
                    likelihood = jnp.broadcast_to(likelihood, expected_shape)
                likelihood_norm = jnp.sum(
                    likelihood * feature_weights, axis=feature_axes, keepdims=True
                )
                likelihood = likelihood / likelihood_norm
                _, _, eig = eval_chunk(likelihood, prior, h0)
                return None, eig

            _, eig_chunks = jax.lax.scan(one_chunk, None, chunk_indices)
            return jnp.reshape(eig_chunks, (-1,))[:total_designs]

        cache[key] = jax.jit(kernel)
        return cache[key]

    def _get_chunked_marginal_eig_kernel(self, nuisance_axes):
        key = (int(self.design_subgrid), tuple(int(v) for v in nuisance_axes))
        cache = self._compiled_kernels["calculateMarginalEIG_chunked"]
        if key in cache:
            return cache[key]

        setup = self._get_chunk_scan_setup()
        chunk_shape = setup["chunk_shape"]
        total_ndim = setup["total_ndim"]
        design_offset = setup["design_offset"]
        total_designs = setup["total_designs"]
        num_chunks = setup["num_chunks"]
        expected_shape = setup["expected_shape"]
        feature_axes = setup["feature_axes"]
        feature_weights = setup["feature_weights"]
        params_bundle = setup["params_bundle"]
        features_bundle = setup["features_bundle"]
        padded_design_axes = setup["padded_design_axes"]
        eval_chunk = self._get_marginal_eig_kernel(chunk_shape, nuisance_axes)

        def kernel(prior, h0):
            chunk_indices = jnp.arange(num_chunks, dtype=jnp.int32)

            def one_chunk(_, chunk_idx):
                start = chunk_idx * key[0]
                design_axes = {}
                for name, values in padded_design_axes.items():
                    chunk_values = jax.lax.dynamic_slice_in_dim(values, start, key[0])
                    shape = [1] * total_ndim
                    shape[design_offset] = key[0]
                    design_axes[name] = jnp.reshape(chunk_values, tuple(shape))
                designs_bundle = _AxisBundle(design_axes)

                likelihood = self.unnorm_lfunc(
                    params_bundle,
                    features_bundle,
                    designs_bundle,
                    **self.lfunc_args,
                )
                likelihood = jnp.asarray(likelihood)
                if tuple(likelihood.shape) != expected_shape:
                    likelihood = jnp.broadcast_to(likelihood, expected_shape)
                likelihood_norm = jnp.sum(
                    likelihood * feature_weights, axis=feature_axes, keepdims=True
                )
                likelihood = likelihood / likelihood_norm
                eig = eval_chunk(likelihood, prior, h0)
                return None, eig

            _, eig_chunks = jax.lax.scan(one_chunk, None, chunk_indices)
            return jnp.reshape(eig_chunks, (-1,))[:total_designs]

        cache[key] = jax.jit(kernel)
        return cache[key]

    def _get_marginal_eig_kernel(self, subgrid_shape, nuisance_axes):
        interest_axes, group_ids, num_segments, interest_shape = self._get_marginal_grouping(
            nuisance_axes
        )
        key = (
            tuple(int(v) for v in subgrid_shape),
            tuple(int(v) for v in nuisance_axes),
        )
        cache = self._compiled_kernels["calculateMarginalEIG"]
        if key in cache:
            return cache[key]

        feature_shape = tuple(int(v) for v in self.features.shape)
        param_shape = tuple(int(v) for v in self.parameters.shape)
        feature_size = int(math.prod(feature_shape))
        subgrid_size = int(math.prod(subgrid_shape))
        param_size = int(math.prod(param_shape))
        param_weights = jnp.reshape(self._grid_weights(self.parameters), (param_size,))
        feature_weights = jnp.reshape(self._grid_weights(self.features), (feature_size,))

        def kernel(sub_likelihood, prior, h0):
            flat = jnp.reshape(
                sub_likelihood,
                (feature_size, subgrid_size) + param_shape,
            )
            design_first = jnp.swapaxes(flat, 0, 1)

            def one_feature(likelihood_row):
                weighted = likelihood_row * prior
                marginal = jnp.sum(weighted * jnp.reshape(param_weights, param_shape))
                posterior = jnp.where(marginal > 0, weighted / marginal, prior)
                post = jax.ops.segment_sum(
                    jnp.reshape(posterior, (param_size,)) * param_weights,
                    group_ids,
                    num_segments=num_segments,
                ).reshape(interest_shape)
                log2post = jnp.where(post > 0, jnp.log2(post), 0.0)
                ig = h0 + jnp.sum(post * log2post)
                return marginal, ig

            def one_design(feature_rows):
                marginals, ig_values = jax.vmap(one_feature)(feature_rows)
                eig = jnp.sum(feature_weights * marginals * ig_values)
                return eig

            eig_values = jax.vmap(one_design)(design_first)
            eig = jnp.reshape(eig_values, key[0])
            return eig

        cache[key] = jax.jit(kernel)
        return cache[key]

    def _get_posterior_kernel(self, cond_shapes):
        key = tuple(tuple(int(v) for v in shape) for shape in cond_shapes)
        cache = self._compiled_kernels["get_posterior"]
        if key in cache:
            return cache[key]

        param_ndim = len(self.parameters.shape)
        sum_axes = tuple(range(-param_ndim, 0))
        param_weights = self._grid_weights(self.parameters)

        def kernel(likelihood, prior):
            weighted = likelihood * prior
            post_norm = jnp.sum(weighted * param_weights, axis=sum_axes, keepdims=True)
            post = jnp.where(post_norm > 0, weighted / post_norm, weighted)
            post = jnp.where(post_norm == 0, prior, post)
            return post

        cache[key] = jax.jit(kernel)
        return cache[key]

    def calculateEIG(self, prior, debug=False):
        with jax.default_device(self.device):
            self.prior = jnp.asarray(prior)
            self.EIG = jnp.full_like(self.EIG, jnp.nan)
        prior_norm = self.parameters.sum(self.prior)
        if not bool(jnp.allclose(prior_norm, 1.0)):
            raise ValueError("Prior probabilities must sum to 1")

        # Calculate prior entropy in bits (careful with x log x = 0 for x=0).
        log2prior = jnp.where(self.prior > 0, jnp.log2(self.prior), 0)
        self.H0 = float(-self.parameters.sum(self.prior * log2prior))
        h0 = jnp.asarray(self.H0, dtype=self.prior.dtype)
        total_designs = int(math.prod(self.designs.shape))

        if self.num_subgrids > 1 and not debug:
            first_subgrid, _ = next(self.designs.subgrid(self.design_subgrid))
            self.subgrid_shape = first_subgrid.shape
            with GridStack(self.features, first_subgrid, self.parameters):
                first_likelihood = self.likelihood_func(first_subgrid)
                first_kernel = self._get_eig_kernel(self.subgrid_shape)
                marginal, ig, _ = first_kernel(first_likelihood, self.prior, h0)
                self.marginal = marginal
                self.IG = ig

            eig_flat = self._get_chunked_eig_kernel()(self.prior, h0)
            self.EIG = jnp.reshape(eig_flat, self.designs.shape)
            self._initialized = True
            return self.designs.getmax(self.EIG)

        eig_flat = jnp.full((total_designs,), jnp.nan, dtype=self.prior.dtype)

        for i, (s, _) in enumerate(self.designs.subgrid(self.design_subgrid)):
            chunk_size = int(math.prod(s.shape))
            if i == 0:
                # Store first subgrid likelihood shape for describe().
                self.subgrid_shape = s.shape

            with GridStack(self.features, s, self.parameters):
                sub_likelihood = self.likelihood_func(s)
                if debug:
                    assert bool(
                        jnp.allclose(self.features.sum(sub_likelihood), 1.0)
                    ), "likelihood normalization failed"

                kernel = self._get_eig_kernel(self.subgrid_shape)
                sub_likelihood = self._pad_subgrid_values(
                    sub_likelihood, s.shape, self.subgrid_shape
                )
                marginal, ig, eig = kernel(sub_likelihood, self.prior, h0)
                if i == 0:
                    # Store first subgrid arrays for describe().
                    self.marginal = marginal
                    self.IG = ig

                if debug:
                    marginal_ref = self.parameters.sum(sub_likelihood * self.prior)
                    assert bool(
                        jnp.allclose(marginal_ref, marginal)
                    ), "marginal check failed"

                    posterior = self.parameters.normalize(sub_likelihood * self.prior)
                    log2posterior = jnp.where(posterior > 0, jnp.log2(posterior), 0)
                    ig_ref = self.H0 + self.parameters.sum(posterior * log2posterior)
                    ig_ref = jnp.where(marginal_ref > 0, ig_ref, 0)
                    assert bool(jnp.allclose(ig_ref, ig)), "IG check failed"

                start = i * self.design_subgrid
                stop = start + chunk_size
                eig_flat = eig_flat.at[start:stop].set(jnp.ravel(eig)[:chunk_size])

        self.EIG = jnp.reshape(eig_flat, self.designs.shape)
        self._initialized = True
        return self.designs.getmax(self.EIG)

    def likelihood_func(self, s):
        likelihood = self.unnorm_lfunc(
            self.parameters, self.features, s, **self.lfunc_args
        )
        with jax.default_device(self.device):
            likelihood = jnp.asarray(likelihood)
        likelihood = self.features.normalize(likelihood)
        expected_shape = self.features.shape + s.shape + self.parameters.shape
        if tuple(likelihood.shape) != expected_shape:
            likelihood = jnp.broadcast_to(likelihood, expected_shape)
        return likelihood

    def calculateMarginalEIG(self, *nuisance_params):
        if not self._initialized:
            raise RuntimeError("Must call calculateEIG before calculateMarginalEIG")

        invalid_names = [
            name for name in nuisance_params if name not in self.parameters.names
        ]
        if invalid_names:
            raise ValueError(f"Invalid nuisance parameters: {invalid_names}")

        nuisance_axes = tuple(
            idx
            for idx, name in enumerate(self.parameters.names)
            if name in nuisance_params
        )
        _, group_ids, num_segments, interest_shape = self._get_marginal_grouping(
            nuisance_axes
        )
        param_size = int(math.prod(self.parameters.shape))
        param_weights = jnp.reshape(self._grid_weights(self.parameters), (param_size,))
        prior_interest = jax.ops.segment_sum(
            jnp.reshape(self.prior, (param_size,)) * param_weights,
            group_ids,
            num_segments=num_segments,
        ).reshape(interest_shape)
        log2prior = jnp.where(prior_interest > 0, jnp.log2(prior_interest), 0)
        h0 = -jnp.sum(prior_interest * log2prior)

        if self.num_subgrids > 1:
            eig_flat = self._get_chunked_marginal_eig_kernel(nuisance_axes)(
                self.prior, h0
            )
            return jnp.reshape(eig_flat, self.designs.shape)

        with jax.default_device(self.device):
            eig_full = jnp.full_like(self.EIG, jnp.nan)
        eig_flat = jnp.full((int(math.prod(self.designs.shape)),), jnp.nan, dtype=self.prior.dtype)
        for i, (s, _) in enumerate(self.designs.subgrid(self.design_subgrid)):
            chunk_size = int(math.prod(s.shape))
            with GridStack(self.features, s, self.parameters):
                sub_likelihood = self.likelihood_func(s)
                kernel = self._get_marginal_eig_kernel(self.subgrid_shape, nuisance_axes)
                sub_likelihood = self._pad_subgrid_values(
                    sub_likelihood, s.shape, self.subgrid_shape
                )
                eig = kernel(sub_likelihood, self.prior, h0)
                start = i * self.design_subgrid
                stop = start + chunk_size
                eig_flat = eig_flat.at[start:stop].set(jnp.ravel(eig)[:chunk_size])
        eig_full = jnp.reshape(eig_flat, self.designs.shape)
        return eig_full

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
        cond_shapes = (cond_features.shape, cond_designs.shape)
        kernel = self._get_posterior_kernel(cond_shapes)

        with jax.default_device(self.device):
            likelihood = jnp.asarray(
                self.unnorm_lfunc(
                    self.parameters,
                    cond_features,
                    cond_designs,
                    **self.lfunc_args,
                )
            )
            posterior = kernel(likelihood, self.prior)
        return posterior

    def update(self, **design_and_features):
        post = self.get_posterior(**design_and_features)
        return self.calculateEIG(post)
