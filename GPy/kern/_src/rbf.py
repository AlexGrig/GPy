# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import weave
from ...util.misc import param_to_array
from stationary import Stationary
from GPy.util.caching import Cache_this
from ...core.parameterization import variational
from psi_comp import ssrbf_psi_comp
from psi_comp.ssrbf_psi_gpucomp import PSICOMP_SSRBF
from ...util.config import *

class RBF(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """
    _support_GPU = True
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False):
        super(RBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        self.weave_options = {}
        self.group_spike_prob = False
        
        if self.useGPU:
            self.psicomp = PSICOMP_SSRBF()
            

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            if self.useGPU:
                return self.psicomp.psicomputations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)[0]
            else:
                return ssrbf_psi_comp.psicomputations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)[0]
        else:
            return self.Kdiag(variational_posterior.mean)

    def psi1(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            if self.useGPU:
                return self.psicomp.psicomputations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)[1]
            else:
                return ssrbf_psi_comp.psicomputations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)[1]
        else:
            _, _, _, psi1 = self._psi1computations(Z, variational_posterior)
        return psi1

    def psi2(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            if self.useGPU:
                return self.psicomp.psicomputations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)[2]
            else:
                return ssrbf_psi_comp.psicomputations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)[2]
        else:
            _, _, _, _, psi2 = self._psi2computations(Z, variational_posterior)
        return psi2

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        # Spike-and-Slab GPLVM
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            if self.useGPU:
                self.psicomp.update_gradients_expectations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variance, self.lengthscale, Z, variational_posterior)
            else:
#                 dL_dvar, dL_dlengscale, dL_dZ, dL_dgamma, dL_dmu, dL_dS = ssrbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variance, self.lengthscale, Z, variational_posterior)
                dL_dvar, dL_dlengscale, _, _, _, _ = ssrbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variance, self.lengthscale, Z, variational_posterior)
                self.variance.gradient = dL_dvar
                self.lengthscale.gradient = dL_dlengscale
                
#                 _, _dpsi1_dvariance, _, _, _, _, _dpsi1_dlengthscale = ssrbf_psi_comp._psi1computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
#                 _, _dpsi2_dvariance, _, _, _, _, _dpsi2_dlengthscale = ssrbf_psi_comp._psi2computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
#     
#                 #contributions from psi0:
#                 self.variance.gradient = np.sum(dL_dpsi0)
#     
#                 #from psi1
#                 self.variance.gradient += np.sum(dL_dpsi1 * _dpsi1_dvariance)
#                 if self.ARD:
#                     self.lengthscale.gradient = (dL_dpsi1[:,:,None]*_dpsi1_dlengthscale).reshape(-1,self.input_dim).sum(axis=0)
#                 else:
#                     self.lengthscale.gradient = (dL_dpsi1[:,:,None]*_dpsi1_dlengthscale).sum()  
#     
#                 #from psi2
#                 self.variance.gradient += (dL_dpsi2 * _dpsi2_dvariance).sum()
#                 if self.ARD:
#                     self.lengthscale.gradient += (dL_dpsi2[:,:,:,None] * _dpsi2_dlengthscale).reshape(-1,self.input_dim).sum(axis=0)
#                 else:
#                     self.lengthscale.gradient += (dL_dpsi2[:,:,:,None] * _dpsi2_dlengthscale).sum()
                    
        elif isinstance(variational_posterior, variational.NormalPosterior):
            l2 = self.lengthscale**2
            if l2.size != self.input_dim:
                l2 = l2*np.ones(self.input_dim)

            #contributions from psi0:
            self.variance.gradient = np.sum(dL_dpsi0)
            self.lengthscale.gradient = 0.

            #from psi1
            denom, _, dist_sq, psi1 = self._psi1computations(Z, variational_posterior)
            d_length = psi1[:,:,None] * ((dist_sq - 1.)/(self.lengthscale*denom) +1./self.lengthscale)
            dpsi1_dlength = d_length * dL_dpsi1[:, :, None]
            if self.ARD:
                self.lengthscale.gradient += dpsi1_dlength.sum(0).sum(0)
            else:
                self.lengthscale.gradient += dpsi1_dlength.sum()
            self.variance.gradient += np.sum(dL_dpsi1 * psi1) / self.variance
            #from psi2
            S = variational_posterior.variance
            _, Zdist_sq, _, mudist_sq, psi2 = self._psi2computations(Z, variational_posterior)
            if not self.ARD:
                self.lengthscale.gradient += self._weave_psi2_lengthscale_grads(dL_dpsi2, psi2, Zdist_sq, S, mudist_sq, l2).sum()
            else:
                self.lengthscale.gradient += self._weave_psi2_lengthscale_grads(dL_dpsi2, psi2, Zdist_sq, S, mudist_sq, l2)

            self.variance.gradient += 2.*np.sum(dL_dpsi2 * psi2)/self.variance

        else:
            raise ValueError, "unknown distriubtion received for psi-statistics"

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        # Spike-and-Slab GPLVM
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            if self.useGPU:
                return self.psicomp.gradients_Z_expectations(dL_dpsi1, dL_dpsi2, self.variance, self.lengthscale, Z, variational_posterior)
            else:
                _, _, dL_dZ, _, _, _ = ssrbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variance, self.lengthscale, Z, variational_posterior)
                return dL_dZ
                
#                 _, _, _, _, _, _dpsi1_dZ, _ = ssrbf_psi_comp._psi1computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
#                 _, _, _, _, _, _dpsi2_dZ, _ = ssrbf_psi_comp._psi2computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
#     
#                 #psi1
#                 grad = (dL_dpsi1[:, :, None] * _dpsi1_dZ).sum(axis=0)
#     
#                 #psi2
#                 grad += (dL_dpsi2[:, :, :, None] * _dpsi2_dZ).sum(axis=0).sum(axis=1)
#     
#                 return grad

        elif isinstance(variational_posterior, variational.NormalPosterior):
            l2 = self.lengthscale **2

            #psi1
            denom, dist, dist_sq, psi1 = self._psi1computations(Z, variational_posterior)
            grad = np.einsum('ij,ij,ijk,ijk->jk', dL_dpsi1, psi1, dist, -1./(denom*l2))

            #psi2
            Zdist, Zdist_sq, mudist, mudist_sq, psi2 = self._psi2computations(Z, variational_posterior)
            term1 = Zdist / l2 # M, M, Q
            S = variational_posterior.variance
            term2 = mudist / (2.*S[:,None,None,:] + l2) # N, M, M, Q

            grad += 2.*np.einsum('ijk,ijk,ijkl->kl', dL_dpsi2, psi2, term1[None,:,:,:] + term2)

            return grad
        else:
            raise ValueError, "unknown distriubtion received for psi-statistics"

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        # Spike-and-Slab GPLVM
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            if self.useGPU:
                return self.psicomp.gradients_qX_expectations(dL_dpsi1, dL_dpsi2, self.variance, self.lengthscale, Z, variational_posterior)
            else:
                _, _, _, dL_dmu, dL_dS, dL_dgamma = ssrbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variance, self.lengthscale, Z, variational_posterior)
                return dL_dmu, dL_dS, dL_dgamma

        elif isinstance(variational_posterior, variational.NormalPosterior):

            l2 = self.lengthscale **2
            #psi1
            denom, dist, dist_sq, psi1 = self._psi1computations(Z, variational_posterior)
            tmp = psi1[:, :, None] / l2 / denom
            grad_mu = np.sum(dL_dpsi1[:, :, None] * tmp * dist, 1)
            grad_S = np.sum(dL_dpsi1[:, :, None] * 0.5 * tmp * (dist_sq - 1), 1)
            #psi2
            _, _, mudist, mudist_sq, psi2 = self._psi2computations(Z, variational_posterior)
            S = variational_posterior.variance
            tmp = psi2[:, :, :, None] / (2.*S[:,None,None,:] + l2)
            grad_mu += -2.*np.einsum('ijk,ijkl,ijkl->il', dL_dpsi2, tmp , mudist)
            grad_S += np.einsum('ijk,ijkl,ijkl->il', dL_dpsi2 , tmp , (2.*mudist_sq - 1))

        else:
            raise ValueError, "unknown distriubtion received for psi-statistics"

        return grad_mu, grad_S

    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    @Cache_this(limit=1)
    def _psi1computations(self, Z, vp):
        mu, S = vp.mean, vp.variance
        l2 = self.lengthscale **2
        denom = S[:, None, :] / l2 + 1. # N,1,Q
        dist = Z[None, :, :] - mu[:, None, :] # N,M,Q
        dist_sq = np.square(dist) / l2 / denom # N,M,Q
        exponent = -0.5 * np.sum(dist_sq + np.log(denom), -1)#N,M
        psi1 = self.variance * np.exp(exponent) # N,M
        return denom, dist, dist_sq, psi1


    @Cache_this(limit=1, ignore_args=(0,))
    def _Z_distances(self, Z):
        Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
        Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
        return Zhat, Zdist

    @Cache_this(limit=1)
    def _psi2computations(self, Z, vp):
        from ..cython import kernels as c_kernels
        
        #if config.getboolean('parallel', 'openmp'):
        #    pragma_string = '#pragma omp parallel for private(tmp, exponent_tmp)'
        #    header_string = '#include <omp.h>'
        #    libraries = ['gomp']
        #else:
        #    pragma_string = ''
        #    header_string = ''
        #    libraries = []

        mu, S = param_to_array(vp.mean), param_to_array(vp.variance)

        N, Q = mu.shape
        M = Z.shape[0]

        #compute required distances
        Zhat, Zdist = self._Z_distances(Z)
        Zdist_sq = np.square(Zdist / self.lengthscale) # M,M,Q

        #allocate memory for the things we want to compute
        mudist = np.empty((N, M, M, Q))
        mudist_sq = np.empty((N, M, M, Q))
        psi2 = np.empty((N, M, M))

        l2 = self.lengthscale **2
        denom = (2.*S[:,None,None,:] / l2) + 1. # N,Q
        half_log_denom = 0.5 * np.log(denom[:,0,0,:])
        denom_l2 = denom[:,0,0,:]*l2

        variance_sq = float(np.square(self.variance))
        
        c_kernels.rbf_psi2(N, M, Q, variance_sq,
                           mu, Zhat, Zdist_sq,
                           mudist, mudist_sq, denom_l2,
                           half_log_denom, psi2)

        return Zdist, Zdist_sq, mudist, mudist_sq, psi2

    def _weave_psi2_lengthscale_grads(self, dL_dpsi2, psi2, Zdist_sq, S, mudist_sq, l2):
        #here's the einsum equivalent, it's ~3 times slower
        #return 2.*np.einsum( 'ijk,ijk,ijkl,il->l', dL_dpsi2, psi2, Zdist_sq * (2.*S[:,None,None,:]/l2 + 1.) + mudist_sq + S[:, None, None, :] / l2, 1./(2.*S + l2))*self.lengthscale

        N, Q = S.shape
        M = psi2.shape[-1]

        S = param_to_array(S)
        result = np.zeros(Q)
        
        from ..cython import kernels as c_kernels
        
        c_kernels.rbf_psi2_lengthscale_grads(N, M, Q,
                                             S, Zdist_sq,
                                             mudist_sq, dL_dpsi2,
                                             psi2, l2,
                                             result)
        
        return 2.*result*self.lengthscale
