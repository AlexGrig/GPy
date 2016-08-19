# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from .. import likelihoods
from GPy.core.model import Model

#from ..inference.latent_function_inference import ss_sparce_inference as ss
from GPy.inference.latent_function_inference import ss_sparse_inference as ss

class SparcePrecisionGP(GP):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf
    :param Norm normalizer: [False]
    :param noise_var: the noise variance for Gaussian likelhood, defaults to 1.

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, noise_var=1.0, name='StateSpaceSparse'):
        
        Model.__init__(self, name) # Call model init. Need to skip call to GP

        if len(X.shape) == 1:
            X = np.atleast_2d(X).T
        self.num_data, self.input_dim = X.shape

        if len(Y.shape) == 1:
            Y = np.atleast_2d(Y).T

        assert self.input_dim==1, "State space methods are only for 1D data"

        if len(Y.shape)==2:
            num_data_Y, self.output_dim = Y.shape
            ts_number = None
        elif len(Y.shape)==3:
            num_data_Y, self.output_dim, ts_number = Y.shape

        self.ts_number = ts_number

        assert num_data_Y == self.num_data, "X and Y data don't match"
        assert self.output_dim == 1, "State space methods are for single outputs only"

        # Make sure the observations are ordered in time
        sort_index = np.argsort(X[:,0])
        self.X = X[sort_index,:]
        self.Y = Y[sort_index,:]

        # Noise variance
        self.likelihood = likelihoods.Gaussian(variance=noise_var)

        # Default kernel
        if kernel is None:
            raise ValueError("State-Space Model: the kernel must be provided.")
        else:
            self.kern = kernel
        # Assert that the kernel is supported
        if not hasattr(self.kern, 'sde'):
            raise NotImplementedError('SDE must be implemented for the kernel being used')            
            
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)
        self.posterior = None

        # Two variables for storing the calls to marginal likelihood function
        # Needed in order to not call twice sparse matrix construction functions.
        self._mll_call_tuple = None
        self._mll_call_dict = None


    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        
        log_marginal_ll, d_log_marginal_ll, self._mll_call_tuple, self._mll_call_dict \
            = ss.SparsePrecision1DInference.inference(self.kern, 
                                                   self.X, self.likelihood, self.Y)
        
        self._log_marginal_likelihood = log_marginal_ll
        self.likelihood.update_gradients(d_log_marginal_ll[-1,0])
        self.kern.sde_update_gradient_full(d_log_marginal_ll[:-1,0])
        
    def _raw_predict(self, Xnew=None):
        """
        
        """
        
        assert Xnew == None, "Currently works only for training data"
       
        Kip = self._mll_call_dict['Kip']
        matrix_blocks = self._mll_call_dict['matrix_blocks']       
        
        mean, var = ss.sparse_inference.mean_var_calc(*self._mll_call_tuple, 
                    Kip=Kip, matrix_blocks=matrix_blocks, matrix_blocks_derivatives=None,
                    compute_inv_main_diag=True)
                    
        return mean, var

    def predict(self, Xnew, include_likelihood=True):
        mu, var = self._raw_predict(Xnew)
        
        if include_likelihood:
            var += self.likelihood.variance
            
        return mu, var     
        
        
        
        