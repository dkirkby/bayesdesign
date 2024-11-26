import numpy as np

from bed.grid import Grid, GridStack


class ExperimentDesigner:
    """Brute force calculation of expected information gain using Grids to define the parameters, features, and designs."""

    def __init__(self, parameters, features, designs, unnorm_lfunc, lfunc_args={}, mem=None):
        """Initialize an experiment designer.

        Parameters
        ----------
        parameters : Grid
            grid used to tabulate the parameter space
        features : Grid
            grid used to tabulate the feature space
        designs : Grid
            grid used to tabulate the design space
        unnorm_lfunc : function
            unnormalized likelihood function that takes the parameters, features,
            and designs as arguments
        lfunc_args : dict
            additional parameters that are required to evaluate the likelihood function
        mem : float
            memory limit in MB
        """
        self.parameters = parameters
        self.features = features
        self.designs = designs
        self.unnorm_lfunc = unnorm_lfunc
        self.lfunc_args = lfunc_args
        if mem is None:
            self.design_subgrid = int(np.prod(self.designs.shape))
            self.num_subgrids = 1
        else:
            if mem <= 0:
                raise ValueError("Memory limit must be positive")
            # Calculate the fractional decrease required to meet the memory limit
            # accounting for the fact that the buffer doubles the memory usage.
            frac = mem / (2 * (np.prod(self.features.shape) * 
                np.prod(self.designs.shape) * 
                np.prod(self.parameters.shape) * 8)/(1 << 20))
            self.design_subgrid = int(frac * np.prod(self.designs.shape))
            self.num_subgrids = np.ceil(np.prod(self.designs.shape) / self.design_subgrid)
            if self.design_subgrid == 0:
                raise ValueError("Memory limit too low,", f"invalid subgrid size: {frac * np.prod(self.designs.shape)} < 1")
        self._initialized = False
        self.EIG = np.full(self.designs.shape, np.nan)

    def calculateEIG(self, prior, debug=False):
        """Calculate the expected information gain for all possible designs.

        Calculates following outputs as attributes of this class:
        - H0: prior entropy in bits
        - IG: grid of information gains in bits over the stack (features, designs).
        - marginal: grid of feature probabilities P(D|xi) marginalized over
          posterior probabilities, defined on the stack (features, designs).
          Normalized over features for each point in the (features, designs) grid.
        - EIG: grid of expected information gain in bits over the designs grid.

        Parameters
        ----------
        prior : numpy array
            Grid of prior probabilities P(theta) over the parameters.
            Must be normalized over parameters.
        debug : bool
            Cross-checked buffer calculations against simpler expressions.
            Note that this uses more memory and will take longer.

        Returns
        -------
        dict of design variable names and corresponding values where the calculated
        EIG is maximized.
        """
        self.prior = prior
        if not np.allclose(self.parameters.sum(self.prior), 1):
            raise ValueError("Prior probabilities must sum to 1")
        # Calculate prior entropy in bits (careful with x log x = 0 for x=0).
        log2prior = np.log2(prior, out=np.zeros_like(prior), where=prior > 0)
        self.H0 = -self.parameters.sum(prior * log2prior)
        for i, (s, mask) in enumerate(self.designs.subgrid(self.design_subgrid)):
            if i == 0:
                # Store first subgrid likelihood for describe function
                self.subgrid_shape = s.shape
            with GridStack(self.features, s, self.parameters):
                sub_likelihood = self.likelihood_func(s) # use of memory
                assert np.allclose(self.features.sum(sub_likelihood), 1)
                # Tabulate the marginal probability P(y|xi) by integrating P(y|theta,xi) P(theta) over theta.
                # No explicit normalization is required since P(y|theta,xi) and P(theta) are already normalized.
                # Check that the likelihood is normalized over features
                self._buffer = np.array(sub_likelihood)
                self._buffer *= self.prior
                if i == 0:
                    # Store first subgrid likelihood for describe function
                    self.marginal = self.parameters.sum(self._buffer)
                marginal = self.parameters.sum(self._buffer)
                if debug:
                    assert np.allclose(self.parameters.sum(self.likelihood_func(s) * self.prior), 
                                    marginal), "marginal check failed"
                # Tabulate the posterior P(theta|y,xi) by normalizing P(y|theta,xi) P(theta) over parameters.
                # Use the prior for any (design, feature) points where the likelihood x prior is zero
                # so the corresponding information gain is zero.
                post_norm = self.parameters.sum(self._buffer, keepdims=True)
                self._buffer = np.divide(
                    self._buffer, post_norm, out=self._buffer, where=post_norm > 0
                )
                if debug:
                    # This will fail if the likelihood*prior is zero for any (design, feature) point.
                    posterior = self.parameters.normalize(self.likelihood_func(s) * self.prior)
                    assert np.allclose(posterior, self._buffer), "posterior check failed"
                # Tabulate the information gain in bits IG = post * log2(post) + H0.
                # Do the calculations in stages to avoid allocating any large temporary arrays.
                np.log2(self._buffer, out=self._buffer, where=self._buffer > 0)
                self._buffer *= self.likelihood_func(s)
                self._buffer *= self.prior
                self._buffer = np.divide(
                    self._buffer, post_norm, out=self._buffer, where=post_norm > 0
                )
                if i == 0:
                    self.IG = self.H0 + self.parameters.sum(self._buffer)
                IG = self.H0 + self.parameters.sum(self._buffer)
                IG[post_norm.reshape(IG.shape) == 0] = 0
                if debug:
                    log2posterior = np.log2(
                        posterior, out=np.zeros_like(posterior), where=posterior > 0
                    )
                    assert np.allclose(self.H0 + self.parameters.sum(posterior * log2posterior), 
                                    IG), "IG check failed"

                self.EIG[mask] = self.features.sum(marginal * IG).flatten()
        self._initialized = True
        del self._buffer
        return self.designs.getmax(self.EIG)

    def likelihood_func(self, s):
        likelihood = self.unnorm_lfunc(self.parameters, self.features, s, **self.lfunc_args)
        self.features.normalize(likelihood)
        return likelihood

    def calculateMarginalEIG(self, *nuisance_params):
        """Calculate the EIG using a posterior that is marginalized over the specified nuisance parameters."""
        if not self._initialized:
            raise RuntimeError("Must call calculateEIG before calculateMarginalEIG")
        # Calculate the marginal prior and its entropy.
        prior = self.parameters.sum(
            self.prior, axis_names=nuisance_params, keepdims=True
        )
        log2prior = np.log2(prior, out=np.zeros_like(prior), where=prior > 0)
        H0 = -self.parameters.sum(prior * log2prior)
        # Calculate the marginal posterior and the information gain.
        EIG = np.full(self.designs.shape, np.nan)
        for (s, mask) in self.designs.subgrid(self.design_subgrid):
            with GridStack(self.features, s, self.parameters):
                sub_likelihood = self.likelihood_func(s)
                self._buffer = np.array(sub_likelihood)
                self._buffer *= self.prior
                marginal = self.parameters.sum(self._buffer)
                post_norm = self.parameters.sum(self._buffer, keepdims=True)
                self._buffer = np.divide(
                    self._buffer, post_norm, out=self._buffer, where=post_norm > 0
                )
                self._buffer = np.add(
                    self._buffer, self.prior, out=self._buffer, where=post_norm == 0
                )
                post = self.parameters.sum(
                    self._buffer, axis_names=nuisance_params, keepdims=True
                )
                # Calculate the information gain for all possible designs and measurements
                log2post = np.log2(post, out=np.zeros_like(post), where=post > 0)
                IG = H0 + self.parameters.sum(post * log2post)
                # Tabulate the expected information gain in bits.
                EIG[mask] = self.features.sum(marginal * IG).flatten()
        del self._buffer
        return EIG

    def describe(self):
        if not self._initialized:
            print("Not initialized")
            return
        for name in ("designs", "features", "parameters"):
            grid = self.__dict__[name]
            if name == "designs" and self.num_subgrids > 1:
                print(f"GRID  {name:>16s} {repr(grid)}, {self.num_subgrids} subgrids used")
            else:
                print(f"GRID  {name:>16s} {repr(grid)}")
        for name in ("prior", "subgrid_shape", "marginal", "IG", "EIG"):
            array = self.__dict__[name]
            if name == "subgrid_shape":
                if self.num_subgrids > 1:
                    full_likelihood_name = "full_likelihood"
                    full_likelihood_shape = str(self.features.shape + self.designs.shape + self.parameters.shape)
                    full_likelihood_size = np.prod(self.features.shape) * np.prod(self.designs.shape) * np.prod(self.parameters.shape) * 8 / (1 << 20)
                    print(
                        f"ARRAY {full_likelihood_name:>16s} {full_likelihood_shape:26s} {full_likelihood_size:9.1f} Mb"
                        )
                name = "likelihood"
                likelihood_shape = str(self.features.shape + self.subgrid_shape + self.parameters.shape)
                likelihood_size = np.prod(self.features.shape) * np.prod(self.subgrid_shape) * np.prod(self.parameters.shape) * 8 / (1 << 20)
                print(
                    f"ARRAY {name:>16s} {likelihood_shape:26s} {likelihood_size:9.1f} Mb"
                    )
            else:
                print(
                    f"ARRAY {name:>16s} {repr(array.shape):26s} {array.nbytes/(1<<20):9.1f} Mb"
                    )


    def get_posterior(self, **design_and_features):
        """Return the posterior P(theta|D,xi) for the specified design and features."""
        if not self._initialized:
            print("Not initialized")
            return
        # match the input names to the grid names
        designs = {}
        features = {}
        for name in design_and_features:
            if name in self.designs.names:
                designs[name] = design_and_features[name]
            elif name in self.features.names:
                features[name] = design_and_features[name]
        cond_designs = Grid(**designs)
        cond_features = Grid(**features)

        self._buffer = self.unnorm_lfunc(self.parameters, cond_features, cond_designs, **self.lfunc_args)
        self._buffer *= self.prior
        post_norm = self.parameters.sum(self._buffer, keepdims=True)
        self._buffer = np.divide(
            self._buffer, post_norm, out=self._buffer, where=post_norm > 0
        )
        self._buffer = np.add(
            self._buffer, self.prior, out=self._buffer, where=post_norm == 0
        )
        return self._buffer


    def update(self, **design_and_features):
        post = self.get_posterior(**design_and_features)
        return self.calculateEIG(post)
