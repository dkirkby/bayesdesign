import numpy as np

from bed.grid import Grid, GridStack
import psutil


class ExperimentDesigner:
    """Brute force calculation of expected information gain using Grids to define the parameters, features, and designs."""

    def __init__(self, parameters, features, designs, likelihood_func, mem=5000):
        """Initialize an experiment designer.

        Parameters
        ----------
        parameters : Grid
            grid used to tabulate the parameter space
        features : Grid
            grid used to tabulate the feature space
        designs : Grid
            grid used to tabulate the design space
        likelihood : numpy array
            grid of likelihood probabilities P(D|theta,xi) over the stack
            (parameters, features, designs). Must be normalized over features
            for each point on the (parameters, designs) grid.
        mem : int
            memory limit in MB for the size of the likelihood grid
        """
        self.parameters = parameters
        self.features = features
        self.designs = designs
        self.likelihood = likelihood_func
        # Calculate the memory required for the likelihood without tabulating the whole grid
        full_grid_bytes = np.prod(parameters.shape) * np.prod(features.shape) * np.prod(designs.shape) * list(parameters.axes.values())[0].itemsize
        # Calculate fractional decrease in memory required for the design subgrid
        frac = mem * (1 << 20) / full_grid_bytes
        self.sub_designs = designs.subgrid(frac)

        with GridStack(features, list(self.sub_designs.values())[0], parameters):
            sub_array = likelihood_func(features, list(self.sub_designs.values())[0], parameters)
            assert np.allclose(self.features.sum(sub_array), 1)
        if sub_array.shape != features.shape + list(self.sub_designs.values())[0].shape + parameters.shape:
            raise ValueError(
                "Likelihood shape not compatible with [features,designs,parameters]"
            )
            
        # Pre-allocate an internal buffer the same size as the input likelihood
        # that we use to avoid allocating any large arrays in the calculate method.
        #self._buffer = np.zeros_like(likelihood_func(features, list(self.sub_designs.values())[0], parameters))
        #self.posterior = self._buffer
        self._initialized = False
        self.EIG = np.zeros(designs.shape)

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

        for idxs, design in self.sub_designs.items():
            with GridStack(self.features, design, self.parameters):

                # Tabulate the marginal probability P(y|xi) by integrating P(y|theta,xi) P(theta) over theta.
                # No explicit normalization is required since P(y|theta,xi) and P(theta) are already normalized.
                self._buffer = self.likelihood(self.features, design, self.parameters)
                self._buffer *= self.prior
                self.marginal = self.parameters.sum(self._buffer)
                if debug:
                    marginal = self.parameters.sum(self.likelihood(self.features, design, self.parameters) * self.prior)
                    assert np.allclose(marginal, self.marginal), "marginal check failed"

                # Tabulate the posterior P(theta|y,xi) by normalizing P(y|theta,xi) P(theta) over parameters.
                post_norm = self.parameters.sum(self._buffer, keepdims=True)
                self._buffer /= post_norm
                if debug:
                    posterior = self.parameters.normalize(self.likelihood(self.features, design, self.parameters) * self.prior)
                    assert np.allclose(posterior, self._buffer), "posterior check failed"

                # Tabulate the information gain in bits IG = post * log2(post) + H0.
                # Do the calculations in stages to avoid allocating any large temporary arrays.
                np.log2(self._buffer, out=self._buffer, where=self._buffer > 0)
                self._buffer *= self.likelihood(self.features, design, self.parameters)
                self._buffer *= self.prior
                self._buffer /= post_norm
                self.IG = self.H0 + self.parameters.sum(self._buffer)
                if debug:
                    log2posterior = np.log2(
                        posterior, out=np.zeros_like(posterior), where=posterior > 0
                    )
                    IG = self.H0 + self.parameters.sum(posterior * log2posterior)
                    assert np.allclose(IG, self.IG), "IG check failed"

                # Leave the posterior in the buffer.
                self._buffer[:] = self.likelihood(self.features, design, self.parameters)
                self._buffer *= self.prior
                self._buffer /= post_norm

            with GridStack(self.features, design):
                # Tabulate the expected information gain in bits as avg of IG(y,xi) with weights P(y|xi).
                EIG = self.features.sum(self.marginal * self.IG)

            # Create slices to record the EIG for the corresponding sub_design
            start_indices = np.array(design.shape) * np.array(idxs)
            end_indices = np.array(design.shape) * (np.array(idxs) + 1)
            if np.array(list(self.sub_designs.keys())).max() in idxs:
                # Handle the case where the last slice is not the same size as the others
                dims = np.where(np.array(idxs) == np.array(list(self.sub_designs.keys())).max())[0]
                start_indices[dims] = np.array(self.designs.shape)[dims] - np.array(design.shape)[dims]
                end_indices[dims] = np.array(self.designs.shape)[dims]

            slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))

            self.EIG[slices] = EIG

        self._initialized = True
        return self.designs.getmax(self.EIG)

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
        with GridStack(self.features, self.designs, self.parameters):
            post = self.parameters.sum(
                self.posterior, axis_names=nuisance_params, keepdims=True
            )
            # Calculate the information gain for all possible designs and measurements
            log2post = np.log2(post, out=np.zeros_like(post), where=post > 0)
            IG = H0 + self.parameters.sum(post * log2post)
            # Tabulate the expected information gain in bits.
            EIG = self.features.sum(self.marginal * IG)
        return EIG

    def describe(self):
        if not self._initialized:
            print("Not initialized")
            return
        for name in ("designs", "features", "parameters"):
            grid = self.__dict__[name]
            print(f"GRID  {name:>12s} {repr(grid)}")
        for name in ("prior", "likelihood", "posterior", "marginal", "IG", "EIG"):
            array = self.__dict__[name]
            print(
                f"ARRAY {name:>12s} {repr(array.shape):22s} {array.nbytes/(1<<20):7.1f} Mb"
            )

    def get_posterior(self, **design_and_features):
        """Return the posterior P(theta|D,xi) for the specified design and features."""
        # TODO: Check that design and features are fully specified...

        with GridStack(self.features, self.designs, self.parameters) as stack:
            loc = stack.at(**design_and_features)
            # Return a copy so that future changes to our buffer have no side effects.
            return np.array(self.posterior[loc])

    def update(self, **design_and_features):
        post = self.get_posterior(**design_and_features)
        return self.calculateEIG(post)
