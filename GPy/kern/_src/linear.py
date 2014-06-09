# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import weave
from kern import Kern
from ...util.linalg import tdot
from ...util.misc import param_to_array
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
from ...util.caching import Cache_this
from ...core.parameterization import variational
from psi_comp import linear_psi_comp
from ...util.config import *

class Linear(Kern):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^input_dim \sigma^2_i x_iy_i

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param ARD: Auto Relevance Determination. If False, the kernel has only one
                variance parameter \sigma^2, otherwise there is one variance
                parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """

    def __init__(self, input_dim, variances=None, ARD=False, active_dims=None, name='linear'):
        super(Linear, self).__init__(input_dim, active_dims, name)
        self.ARD = ARD
        if not ARD:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
        else:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == self.input_dim, "bad number of variances, need one ARD variance per input_dim"
            else:
                variances = np.ones(self.input_dim)

        self.variances = Param('variances', variances, Logexp())
        self.add_parameter(self.variances)

    @Cache_this(limit=2)
    def K(self, X, X2=None):
        if self.ARD:
            if X2 is None:
                return tdot(X*np.sqrt(self.variances))
            else:
                rv = np.sqrt(self.variances)
                return np.dot(X*rv, (X2*rv).T)
        else:
            return self._dot_product(X, X2) * self.variances

    @Cache_this(limit=1, ignore_args=(0,))
    def _dot_product(self, X, X2=None):
        if X2 is None:
            return tdot(X)
        else:
            return np.dot(X, X2.T)

    def Kdiag(self, X):
        return np.sum(self.variances * np.square(X), -1)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if self.ARD:
            if X2 is None:
                self.variances.gradient = np.array([np.sum(dL_dK * tdot(X[:, i:i + 1])) for i in range(self.input_dim)])
            else:
                product = X[:, None, :] * X2[None, :, :]
                self.variances.gradient = (dL_dK[:, :, None] * product).sum(0).sum(0)
        else:
            self.variances.gradient = np.sum(self._dot_product(X, X2) * dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        tmp = dL_dKdiag[:, None] * X ** 2
        if self.ARD:
            self.variances.gradient = tmp.sum(0)
        else:
            self.variances.gradient = np.atleast_1d(tmp.sum())


    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None:
            return 2.*(((X[None,:, :] * self.variances)) * dL_dK[:, :, None]).sum(1)
        else:
            return (((X2[None,:, :] * self.variances)) * dL_dK[:, :, None]).sum(1)

    def gradients_X_diag(self, dL_dKdiag, X):
        return 2.*self.variances*dL_dKdiag[:,None]*X

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            gamma = variational_posterior.binary_prob
            mu = variational_posterior.mean
            S = variational_posterior.variance

            return np.einsum('q,nq,nq->n',self.variances,gamma,np.square(mu)+S)
#            return (self.variances*gamma*(np.square(mu)+S)).sum(axis=1)
        else:
            return np.sum(self.variances * self._mu2S(variational_posterior), 1)

    def psi1(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            gamma = variational_posterior.binary_prob
            mu = variational_posterior.mean
            return np.einsum('nq,q,mq,nq->nm',gamma,self.variances,Z,mu)
#            return (self.variances*gamma*mu).sum(axis=1)
        else:
            return self.K(variational_posterior.mean, Z) #the variance, it does nothing

    @Cache_this(limit=1)
    def psi2(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            gamma = variational_posterior.binary_prob
            mu = variational_posterior.mean
            S = variational_posterior.variance
            mu2 = np.square(mu)
            variances2 = np.square(self.variances)
            tmp = np.einsum('nq,q,mq,nq->nm',gamma,self.variances,Z,mu)
            return np.einsum('nq,q,mq,oq,nq->nmo',gamma,variances2,Z,Z,mu2+S)+\
                np.einsum('nm,no->nmo',tmp,tmp) - np.einsum('nq,q,mq,oq,nq->nmo',np.square(gamma),variances2,Z,Z,mu2)
        else:
            ZA = Z * self.variances
            ZAinner = self._ZAinner(variational_posterior, Z)
            return np.dot(ZAinner, ZA.T)

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            gamma = variational_posterior.binary_prob
            mu = variational_posterior.mean
            S = variational_posterior.variance
            mu2S = np.square(mu)+S
            _dpsi2_dvariance, _, _, _, _ = linear_psi_comp._psi2computations(self.variances, Z, mu, S, gamma)
            grad = np.einsum('n,nq,nq->q',dL_dpsi0,gamma,mu2S) + np.einsum('nm,nq,mq,nq->q',dL_dpsi1,gamma,Z,mu) +\
                 np.einsum('nmo,nmoq->q',dL_dpsi2,_dpsi2_dvariance)
            if self.ARD:
                self.variances.gradient = grad
            else:
                self.variances.gradient = grad.sum()
        else:
            #psi1
            self.update_gradients_full(dL_dpsi1, variational_posterior.mean, Z)
            # psi0:
            tmp = dL_dpsi0[:, None] * self._mu2S(variational_posterior)
            if self.ARD: self.variances.gradient += tmp.sum(0)
            else: self.variances.gradient += tmp.sum()
            #psi2
            if self.ARD:
                tmp = dL_dpsi2[:, :, :, None] * (self._ZAinner(variational_posterior, Z)[:, :, None, :] * Z[None, None, :, :])
                self.variances.gradient += 2.*tmp.sum(0).sum(0).sum(0)
            else:
                self.variances.gradient += 2.*np.sum(dL_dpsi2 * self.psi2(Z, variational_posterior))/self.variances

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            gamma = variational_posterior.binary_prob
            mu = variational_posterior.mean
            S = variational_posterior.variance
            _, _, _, _, _dpsi2_dZ = linear_psi_comp._psi2computations(self.variances, Z, mu, S, gamma)

            grad =  np.einsum('nm,nq,q,nq->mq',dL_dpsi1,gamma, self.variances,mu) +\
                 np.einsum('nmo,noq->mq',dL_dpsi2,_dpsi2_dZ)

            return grad
        else:
            #psi1
            grad = self.gradients_X(dL_dpsi1.T, Z, variational_posterior.mean)
            #psi2
            self._weave_dpsi2_dZ(dL_dpsi2, Z, variational_posterior, grad)
            return grad

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            gamma = variational_posterior.binary_prob
            mu = variational_posterior.mean
            S = variational_posterior.variance
            mu2S = np.square(mu)+S
            _, _dpsi2_dgamma, _dpsi2_dmu, _dpsi2_dS, _ = linear_psi_comp._psi2computations(self.variances, Z, mu, S, gamma)

            grad_gamma = np.einsum('n,q,nq->nq',dL_dpsi0,self.variances,mu2S) + np.einsum('nm,q,mq,nq->nq',dL_dpsi1,self.variances,Z,mu) +\
                 np.einsum('nmo,nmoq->nq',dL_dpsi2,_dpsi2_dgamma)
            grad_mu = np.einsum('n,nq,q,nq->nq',dL_dpsi0,gamma,2.*self.variances,mu) + np.einsum('nm,nq,q,mq->nq',dL_dpsi1,gamma,self.variances,Z) +\
                 np.einsum('nmo,nmoq->nq',dL_dpsi2,_dpsi2_dmu)
            grad_S = np.einsum('n,nq,q->nq',dL_dpsi0,gamma,self.variances) + np.einsum('nmo,nmoq->nq',dL_dpsi2,_dpsi2_dS)

            return grad_mu, grad_S, grad_gamma
        else:
            grad_mu, grad_S = np.zeros(variational_posterior.mean.shape), np.zeros(variational_posterior.mean.shape)
            # psi0
            grad_mu += dL_dpsi0[:, None] * (2.0 * variational_posterior.mean * self.variances)
            grad_S += dL_dpsi0[:, None] * self.variances
            # psi1
            grad_mu += (dL_dpsi1[:, :, None] * (Z * self.variances)).sum(1)
            # psi2
            self._weave_dpsi2_dmuS(dL_dpsi2, Z, variational_posterior, grad_mu, grad_S)

            return grad_mu, grad_S

    #--------------------------------------------------#
    #            Helpers for psi statistics            #
    #--------------------------------------------------#


    def _weave_dpsi2_dmuS(self, dL_dpsi2, Z, vp, target_mu, target_S):
        # Think N,num_inducing,num_inducing,input_dim
        ZA = Z * self.variances
        AZZA = ZA.T[:, None, :, None] * ZA[None, :, None, :]
        AZZA = AZZA + AZZA.swapaxes(1, 2)
        from ..cython import kernels as c_kernels
        mu = vp.mean
        N,num_inducing,input_dim,mu = mu.shape[0],Z.shape[0],mu.shape[1],param_to_array(mu)
        c_kernels.linear_dpsi2_dmuS(N, num_inducing, input_dim,
                                    mu, AZZA, target_mu, target_S, dL_dpsi2)

    def _weave_dpsi2_dZ(self, dL_dpsi2, Z, vp, target):
        AZA = self.variances*self._ZAinner(vp, Z)

        from ..cython import kernels as c_kernels
        N,num_inducing,input_dim = vp.mean.shape[0],Z.shape[0],vp.mean.shape[1]
        c_kernels.linear_dpsi2_dZ(N, num_inducing, input_dim,
                                  AZA, target, dL_dpsi2)

    @Cache_this(limit=1, ignore_args=(0,))
    def _mu2S(self, vp):
        return np.square(vp.mean) + vp.variance

    @Cache_this(limit=1)
    def _ZAinner(self, vp, Z):
        ZA = Z*self.variances
        inner = (vp.mean[:, None, :] * vp.mean[:, :, None])
        diag_indices = np.diag_indices(vp.mean.shape[1], 2)
        inner[:, diag_indices[0], diag_indices[1]] += vp.variance

        return np.dot(ZA, inner).swapaxes(0, 1)  # NOTE: self.ZAinner \in [num_inducing x num_data x input_dim]!

    def input_sensitivity(self):
        return np.ones(self.input_dim) * self.variances

class LinearFull(Kern):
    def __init__(self, input_dim, rank, W=None, kappa=None, active_dims=None, name='linear_full'):
        super(LinearFull, self).__init__(input_dim, active_dims, name)
        if W is None:
            W = np.ones((input_dim, rank))
        if kappa is None:
            kappa = np.ones(input_dim)
        assert W.shape == (input_dim, rank)
        assert kappa.shape == (input_dim,)

        self.W = Param('W', W)
        self.kappa = Param('kappa', kappa, Logexp())
        self.add_parameters(self.W, self.kappa)

    def K(self, X, X2=None):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return np.einsum('ij,jk,lk->il', X, P, X if X2 is None else X2)

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.kappa.gradient = np.einsum('ij,ik,kj->j', X, dL_dK, X if X2 is None else X2)
        self.W.gradient = np.einsum('ij,kl,ik,lm->jm', X, X if X2 is None else X2, dL_dK, self.W)
        self.W.gradient += np.einsum('ij,kl,ik,jm->lm', X, X if X2 is None else X2, dL_dK, self.W)

    def Kdiag(self, X):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return np.einsum('ij,jk,ik->i', X, P, X)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.kappa.gradient = np.einsum('ij,i->j', np.square(X), dL_dKdiag)
        self.W.gradient = 2.*np.einsum('ij,ik,jl,i->kl', X, X, self.W, dL_dKdiag)

    def gradients_X(self, dL_dK, X, X2=None):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        if X2 is None:
            return 2.*np.einsum('ij,jk,kl->il', dL_dK, X, P)
        else:
            return np.einsum('ij,jk,kl->il', dL_dK, X2, P)

    def gradients_X_diag(self, dL_dKdiag, X):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return 2.*np.einsum('jk,i,ij->ik', P, dL_dKdiag, X)


