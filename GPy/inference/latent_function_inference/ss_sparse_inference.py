# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import PosteriorExact as Posterior
from ...models import state_space_main as ssm
from ... import likelihoods
import warnings

from scipy import sparse
import scipy.linalg as la
#import sksparse.cholmod as cholmod

import time
import numpy as np
import scipy as sp
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)


def solve_ols_svd(U,S,Vh, Y, lamda = 0.0 ):
    """
    Solve OLS problem given the SVD decomposition
    
    Input:
        ! Note X= U*S*Vh and Y are assumed to be normalized, hence lamda is between 0.0 and 1.0.
    
        U, S, Vh - SVD decomposition
        Y - target variables
        lamda - regularization parameter. Lamda must be normalized with respect
                                          to number of samples. Data is assumed
                                          to be normalized, so lamda is between 0.0 and 1.0.
    """
    
    n_points = U.shape[0]
    machine_epsilon = np.finfo(np.float64).eps
    if (lamda is None) or (lamda == 0.0) or ( lamda > 1.0) or ( lamda < 0.0) : # no lamda optimization happened or wrong optim results
        num_rank = np.count_nonzero(S > S[0] * machine_epsilon)  # numerical rank  
    
        S.shape = (S.shape[0],1)
        coeffs = np.dot( Vh.T, np.multiply( 1.0/S ,np.dot( U.T, Y ) ) )
        
    else:
        S2 = np.power(S,2)
        S2 = S2 + n_points*lamda # parameter lambda normalized with respect to number of points !!!
        S = S / S2
        S.shape = (S.shape[0],1)
        coeffs = np.dot( Vh.T, np.multiply( S ,np.dot( U.T, Y ) ) )
        
        num_rank = None # numerical rank is None because regularization is used
    
    return coeffs, num_rank

class SparsePrecision1DInference(LatentFunctionInference):
    """
    Sparce Precision 1D Inference
    """
    def __init__(self):
        pass
    
#    diff_x_crit = 1e-14 # if new X_test points are added, this tells when
#                        # to consider new points to be distinct.     
    
    @staticmethod
    def inference(kernel, X, Y, var_or_likelihood, p_balance=False, p_largest_cond_num=1e+16, p_regularization_type=2):
        """
        Returns marginal likelihood (MLL), MLL gradient, and dicts with variaous input matrices.
        
        Inputs:
        ------------------------
        
        kernel: kernel object
        
        X: array(N,1)
        
        Y: array(N,1)
        
        var_or_likelihood: either likelihood object or float with noise variance.
        
        p_balance: bool
            Balance general mode or not
            
        p_largest_cond_num: float
            Largest condition number of the Qk matices. See function 
            "sparse_inverse_cov".
            
        p_regularization_type: 1 or 2
        
        """
        
        if isinstance(var_or_likelihood, likelihoods.Gaussian):
            noise_var = float(var_or_likelihood.gaussian_variance() )
        elif isinstance(var_or_likelihood, float):
            noise_var = var_or_likelihood
        else:
            raise ValueError("SparsePrecision1DInference.inference: var_or_likelihood has incorrect type.")
        
        #import pdb; pdb.set_trace()
         
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel.sde()
        block_size = F.shape[1]    
        
        if p_balance:
            (F,L,Qc,H,P_inf,P0, dF,dQct,dP_inft,dP0t) = ssm.balance_ss_model(F,L,Qc,H,P_inf,P0, dFt,dQct,dP_inft, dP0t)
            print("SparsePrecision1DInference.inference: Balancing!")
        
        Ki_diag, Ki_low_diag, Ki_logdet, d_Ki_diag, d_Ki_low_diag, \
                   d_Ki_logdet = btd_inference.build_matrices(X, Y, 
                   F, L, Qc, P_inf, P0, H, p_largest_cond_num, p_regularization_type=p_regularization_type, 
                   compute_derivatives=True, dP_inf=dP_inft, dP0 = dP0t, dF=dFt, dQc=dQct)        
        
        (marginal_ll, d_marginal_ll, mll_data_fit_term, mll_log_det, \
        mll_data_fit_deriv, mll_determ_deriv)  = btd_inference.marginal_ll(block_size, Y, Ki_diag, Ki_low_diag, Ki_logdet, H, noise_var, 
                    compute_derivatives=True, d_Ki_diag=d_Ki_diag, d_Ki_low_diag=d_Ki_low_diag,
                    d_Ki_logdet = d_Ki_logdet)
                    
        return marginal_ll, d_marginal_ll, mll_data_fit_term, mll_log_det, mll_data_fit_deriv, mll_determ_deriv       

    @staticmethod
    def mean_and_var(kernel, X_train, Y_train, var_or_likelihood, X_test=None, p_balance=False, p_largest_cond_num=1e+16,
                     p_regularization_type=2, diff_x_crit=None):
        """
        Computes mean and variance of the posterior.
        
        Input:
        ---------------
        Same as for inference        
        
        X_train: array(N,1) 
        
        Y_train: array(N,1) 
        
        X_test: array(M,1) or None
            Test data points.
        
        p_balance: bool
            Balance general mode or not
        
        p_largest_cond_num: float
            Largest condition number of the Qk matices. See function 
            "sparse_inverse_cov".
        
        mll_call_tuple: tuple
            This is tuple is obtained from the "sparse_inverse_cov" function,
            and if X_test = None, then it can be reused here.
        
        mll_call_dict:
            Dict obtained from "sparse_inverse_cov" function.
        
        diff_x_crit: float (not currently used)   
            If new X_test points are added, this tells when to consider 
            new points to be distinct. If it is None then the same variable
            is taken from the class.
            
        p_regularization_type: 1 or 2
        """        
        
        if isinstance(var_or_likelihood, likelihoods.Gaussian):
            noise_var = var_or_likelihood.gaussian_variance()
        elif isinstance(var_or_likelihood, float):
            noise_var = var_or_likelihood
        else:
            raise ValueError("SparsePrecision1DInference.inference: var_or_likelihood has incorrect type.")
        
#        if diff_x_crit is None:
#            diff_x_crit = SparsePrecision1DInference.diff_x_crit
            
        train_points_num = X_train.size
        
        #import pdb; pdb.set_trace()  
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel.sde()
        block_size = F.shape[1]    
        
        if p_balance:
            (F,L,Qc,H,P_inf,P0, dF,dQct,dP_inft,dP0t) = ssm.balance_ss_model(F,L,Qc,H,P_inf,P0, dFt,dQct,dP_inft, dP0t)
            print("SparsePrecision1DInference.mean_and_var: Balancing!")
            
        block_size = F.shape[1]   
        
        Ki_diag, Ki_low_diag, test_points_num, forward_index, inverse_index = \
                btd_inference.mean_var_calc_prepare_matrices(block_size, X_train, X_test, 
                                     Y_train, noise_var, F, L, Qc, P_inf, P0, H,
                                       p_largest_cond_num, p_regularization_type, diff_x_crit=None)
                                       
        sp_mean, sp_var = btd_inference.mean_var_calc(block_size, Y_train, Ki_diag, 
                                                      Ki_low_diag, H, noise_var, test_points_num, forward_index, inverse_index)
        return sp_mean, sp_var
        
class SparsePrecision1DInferenceOld(LatentFunctionInference):
    """
    Sparce Precision 1D Inference
    """
    def __init__(self):
        pass
    
    diff_x_crit = 1e-14 # if new X_test points are added, this tells when
                        # to consider new points to be distinct.     
    
    @staticmethod
    def inference(kernel, X, Y, var_or_likelihood, p_balance=False, p_largest_cond_num=1e+16, p_regularization_type=2):
        """
        Returns marginal likelihood (MLL), MLL gradient, and dicts with variaous input matrices.
        
        
        Inputs:
        ------------------------
        
        kernel: kernel object
        
        X: array(N,1)
        
        Y: array(N,1)
        
        var_or_likelihood: either likelihood object or float with noise variance.
        
        p_balance: bool
            Balance general mode or not
            
        p_largest_cond_num: float
            Largest condition number of the Qk matices. See function 
            "sparse_inverse_cov".
            
        p_regularization_type: 1 or 2
        
        """
        
        if isinstance(var_or_likelihood, likelihoods.Gaussian):
            noise_var = var_or_likelihood.gaussian_variance()
        elif isinstance(var_or_likelihood, float):
            noise_var = var_or_likelihood
        else:
            raise ValueError("SparsePrecision1DInference.inference: var_or_likelihood has incorrect type.")
          
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel.sde()
        block_size = F.shape[1]    
        
        if p_balance:
            (F,L,Qc,H,P_inf,P0, dF,dQct,dP_inft,dP0t) = ssm.balance_ss_model(F,L,Qc,H,P_inf,P0, dFt,dQct,dP_inft, dP0t)
            print("SparsePrecision1DInference.inference: Balancing!")
        
        grad_calc_params = {}
        grad_calc_params['dP_inf'] = dP_inft
        grad_calc_params['dF'] = dFt
        grad_calc_params['dQc'] = dQct
        
        (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
         matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(X, 
                Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, compute_derivatives=True,
                           grad_calc_params=grad_calc_params,p_regularization_type=p_regularization_type)
        
        mll_call_tuple = (block_size, Y, Ait, Qi, GtY, G, GtG, H, noise_var)        
        mll_call_dict = { 'compute_derivatives':True, 'dKi_vector':Ki_derivatives,
                          'Kip':Kip, 'matrix_blocks':matrix_blocks, 
                          'matrix_blocks_derivatives': matrix_blocks_derivatives }
        
        res = sparse_inference.marginal_ll( *mll_call_tuple, **mll_call_dict)
        
        marginal_ll = res[0] # margianal likelihood 
        d_marginal_ll = res[1] # margianal likelihood  gradient
        del res
        #tmp_inv_data = res[2] # extra data for inversion
        
        #test_inverse = sparse_inference.sparse_inv_rhs(n_points, block_size, matrix_blocks, HtH/noise_var, tridiag_inv_data, rhs_block)
        return marginal_ll, d_marginal_ll, mll_call_tuple, mll_call_dict       

    @staticmethod
    def mean_and_var(kernel, X_train, Y_train, var_or_likelihood, X_test=None, p_balance=False, p_largest_cond_num=1e+16,
                     p_regularization_type=2, mll_call_tuple=None, mll_call_dict=None, diff_x_crit=None):
        """
        Computes mean and variance of the posterior.
        
        Input:
        ---------------
        Same as for inference        
        
        X_train: array(N,1) 
        
        Y_train: array(N,1) 
        
        X_test: array(M,1) or None
            Test data points.
        
        p_balance: bool
            Balance general mode or not
        
        p_largest_cond_num: float
            Largest condition number of the Qk matices. See function 
            "sparse_inverse_cov".
        
        mll_call_tuple: tuple
            This is tuple is obtained from the "sparse_inverse_cov" function,
            and if X_test = None, then it can be reused here.
        
        mll_call_dict:
            Dict obtained from "sparse_inverse_cov" function.
        
        diff_x_crit: float (not currently used)   
            If new X_test points are added, this tells when to consider 
            new points to be distinct. If it is None then the same variable
            is taken from the class.
            
        p_regularization_type: 1 or 2
        """        
        
        if isinstance(var_or_likelihood, likelihoods.Gaussian):
            noise_var = var_or_likelihood.gaussian_variance()
        elif isinstance(var_or_likelihood, float):
            noise_var = var_or_likelihood
        else:
            raise ValueError("SparsePrecision1DInference.inference: var_or_likelihood has incorrect type.")
        
        if diff_x_crit is None:
            diff_x_crit = SparsePrecision1DInference.diff_x_crit
            
        train_points_num = X_train.size
        
        #import pdb; pdb.set_trace()  
        if (X_test is None) or (X_test is X_train) or ( (X_test.shape == X_train.shape) and np.all(X_test == X_train) ):
            # Consider X_test is equal to X_train
            X_test = None            
            
        if X_test is not None:
            X = np.vstack((X_train, X_test))
            Y = np.vstack((Y_train, np.zeros(X_test.shape)) )
            
            which_test = np.vstack( ( np.ones( X_train.shape), np.zeros( X_test.shape)) )
            
            _, return_index, return_inverse = np.unique(X,True,True)
             
            X = X[return_index]
            Y = Y[return_index]
            which_test = which_test[return_index]
            predict_only_training = False
        else:
            X = X_train
            Y = Y_train
            which_test = None
            predict_only_training = True
            
        if (not predict_only_training) or (mll_call_tuple is None):
            (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel.sde()
            block_size = F.shape[1]    
            
            if p_balance:
                (F,L,Qc,H,P_inf,P0, dF,dQct,dP_inft,dP0t) = ssm.balance_ss_model(F,L,Qc,H,P_inf,P0, dFt,dQct,dP_inft, dP0t)
                print("SparsePrecision1DInference.mean_and_var: Balancing!")
            
#            grad_calc_params = {}
#            grad_calc_params['dP_inf'] = dP_inft
#            grad_calc_params['dF'] = dFt
#            grad_calc_params['dQc'] = dQct
            
            (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
             matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(X, 
                    Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, compute_derivatives=False,
                               grad_calc_params=None, p_regularization_type=p_regularization_type )
                           
            mll_call_tuple = (block_size, Y, Ait, Qi, GtY, G, GtG, H, noise_var)
            
#            def sparse_inverse_cov(X, Y, F, L, Qc, P_inf, P0, H, compute_derivatives=False,
#                           grad_calc_params=None)
        else:
            matrix_blocks = mll_call_dict['matrix_blocks']
            
        #import pdb; pdb.set_trace()                   
        mean, var = sparse_inference.mean_var_calc(*mll_call_tuple, 
                        matrix_blocks, which_observed=which_test, inv_precomputed=None)
        
        if (not predict_only_training):    
            mean = mean[return_inverse]
            var = var[return_inverse]            
                        
            mean = mean[train_points_num:]
            var = var[train_points_num:]
            
        return mean, var, mll_call_tuple
        
#class sparse_inference(object):
#    
#    @staticmethod
#    #@profile
#    def sparse_inverse_cov(X, Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, compute_derivatives=False,
#                           grad_calc_params=None, p_regularization_type=2):
#        """
#        Function returns all the necessary matrices for the SpInGP inference.        
#        
#         Notation for matrix is: K^{-1} = A^{-T} Q^{-1} A^{-1} = [ A0, B1, 0       
#                                                                   C1, A1, B2
#                                                                   0 , C2, A2 ]
#                                    
#        Input:
#        ------------------------------
#            Exactly from furnction kern.sde()
#            
#            H - row vector
#            
#            p_largest_cond_num: float
#            Largest condition number of the Qk and P_inf matices. This is needed because
#            for some models the these matrices are while inverse is required.
#                
#        Output:
#        -------------------------------
#        
#        Ait: sparse(Ai_size, Ai_size)
#            A^{-T} matrix
#        Qi: sparse(Ai_size, Ai_size)
#            Q^{-1} matrix
#            
#        GtY: help matrix 1 
#        G: help matrix 2 
#        GtG: help matrix 3
#        HtH: help matrix 4
#        
#        Ki_derivatives,
#        
#        Kip: sparse(Ai_size, Ai_size)
#            Block tridiagonal matrix. The tridiagonal blocks are the same as in K (inverse of Ki)
#            Required for ll derivarive computation.
#            TODO: note, because it is computed reqursively there may be a loss of accuracy!
#            
#        matrix_blocks: dictionary
#            Contains 2 keys: 'Akd' and 'Ckd'. Another form of writing matrix K^{-1} = Ki.
#            Look their derivatives for shapes.
#            
#        matrix_blocks_derivatives: dictionary
#            Dictionary contains 2 keys: 'dAkd' and 'dCkd'. Each contain corresponding
#                derivatives of matrices Ak and Ck. Shapes dAkd[0:(N-1)] [0:(deriv_num-1),0:(block_size-1), 0:(block_size-1)]
#                    dCkd[0:(N-2)][0:(deriv_num-1), 0:(block_size-1), 0:(block_size-1)]
#        
#        """
#        block_size = F.shape[0]
#        x_points_num = X.shape[0]
#        
#        Ai_size = x_points_num * block_size
#        Ait = sparse.lil_matrix( (Ai_size, Ai_size))
#        Qi = sparse.lil_matrix( (Ai_size, Ai_size) )
#        
#        if not isinstance(p_largest_cond_num, float):
#            raise ValueError('sparse_inference.sparse_inverse_cov: p_Inv_jitter is not float!')
#        
#        Akd = {}
#        Bkd = {}
#        if compute_derivatives:
#            # In this case compute 2 extra matrices dA/d(Theta) and dQ^{-1}/d(Theta)
#        
#            dP_inf = grad_calc_params['dP_inf']
#            dF = grad_calc_params['dF']
#            dQc = grad_calc_params['dQc']
#            
#            # The last parameter is the noice variance
#            grad_params_no = dF.shape[2]
#            
#            
#            #dP0 = grad_calc_params['dP_init']
#            
#            AQcomp = ssm.ContDescrStateSpace._cont_to_discrete_object(X, F, L, Qc, compute_derivatives=True,
#                                     grad_params_no=grad_params_no,
#                                     P_inf=P_inf, dP_inf=dP_inf, dF = dF, dQc=dQc,
#                                     dt0='no dt0')
#            
#            Kip = sparse.lil_matrix( (Ai_size, Ai_size) )                         
#            # Derivative matrices                                     
#            Ait_derivatives = []
#            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
#                Ait_derivatives.append( sparse.lil_matrix( (Ai_size, Ai_size)) )
#                
#            Qi_derivatives = []                      
#            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
#                Qi_derivatives.append( sparse.lil_matrix( (Ai_size, Ai_size)) )
#            
#            Ki_derivatives = [] # Derivatives of the K = At*Qi*A
#
#            # Determinant derivatives: (somematrices required for speeding up the computation of determ derivatives)
#            dAkd = {}
#            dCkd = {}            
#        else:
#            
#            AQcomp = ssm.ContDescrStateSpace._cont_to_discrete_object(X, F, L, Qc, compute_derivatives=False,
#                                     grad_params_no=None,
#                                     P_inf=P_inf, dP_inf=None, dF = None, dQc=None,
#                                     dt0='no dt0')
#            Ait_derivatives = None
#            Qi_derivatives = None
#            Ki_derivatives = None
#            Kip = None
#            
#        b_ones = np.eye(block_size)
#        
#        #GtY = sparse.lil_matrix((Ai_size,1))
#        H_nonzero_inds = np.nonzero(H)[1]
#     
#        G = sparse.kron( sparse.eye(x_points_num, format='csc' ), sparse.csc_matrix(H), format='csc' )
#        HtH = np.dot(H.T, H)
#        GtG = sparse.kron( sparse.eye(x_points_num, format='csc' ), sparse.csc_matrix( HtH ), format='csc')    
#        
#        Ait[0:block_size,0:block_size] = b_ones
#        #import pdb; pdb.set_trace()
#        P_inf = 0.5*(P_inf + P_inf.T)
#                
#        #p_regularization_type=1
#        #p_largest_cond_num = 1e8
#        (U,S,Vh) = la.svd(P_inf, compute_uv=True,)
#        P_inf_inv = ssm.psd_matrix_inverse(-1, P_inf, U=U,S=S,p_largest_cond_num=p_largest_cond_num, regularization_type=p_regularization_type)
#        
#        # Different inverse computation >-
#                    
#        Qi[0:block_size,0:block_size] = P_inf_inv
#        Qk_inv_prev = P_inf_inv # required for derivative of determinant computations
#        
#        if compute_derivatives:
#            d_Qk_inv_prev = np.empty( (grad_params_no, block_size, block_size) ) # initial derivatives of dQk_inv
#            
#            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
#                    dP_inf_p = dP_inf[:,:,dd]            
#                    d_Qk_inv_prev[dd,:,:] = -np.dot(P_inf_inv, np.dot(dP_inf_p, P_inf_inv))
#                    
#        for k in range(0,x_points_num-1):
#            Ak = AQcomp.Ak(k,None,None)
#            Qk = AQcomp.Qk(k)
#            Qk = 0.5*(Qk + Qk.T) # symmetrize because if Qk is not full rank it becomes not symmetric due to numerical problems
#            #import pdb; pdb.set_trace()
#            Qk_inv = AQcomp.Q_inverse(k, p_largest_cond_num, p_regularization_type) # in AQcomp numbering starts from 0, consequence of Python indexing.
#            if np.any((np.abs(Qk_inv - Qk_inv.T)) > 0):
#                raise ValueError('sparse_inverse_cov: Qk_inv is not symmetric!')
#            
#            row_ind_start = (k+1)*block_size
#            row_ind_end = row_ind_start + block_size
#            col_ind_start = k*block_size
#            col_ind_end = col_ind_start + block_size
#            
#            Ait[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = -Ak.T
#            Ait[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = b_ones        
#            Qi[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = Qk_inv
#            
#            #import pdb; pdb.set_trace()
#            #GtY[row_ind_start + H_nonzero_inds,0] = Y[k+1,0]* Hnn     
#            Bkd[k] = - np.dot( Ak.T, Qk_inv )
#            Akd[k] = np.dot( Ak.T, np.dot( Qk_inv, Ak ) ) +  Qk_inv_prev
#                
#            if compute_derivatives:
#                if (k == 0):
#                    prev_Ak = Ak # Ak from the previous step
#                    prev_diag = P_inf
#                    prev_off_diag = np.dot(Ak, P_inf) # previous off diagonal (lowe part)                    
#                    
#                    Kip[col_ind_start:col_ind_end, col_ind_start:col_ind_end] = prev_diag
#                    Kip[row_ind_start:row_ind_end, col_ind_start:col_ind_end] = prev_off_diag
#                    Kip[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = prev_off_diag.T
#                    
#                    prev_diag = np.dot( Ak, np.dot(prev_diag, prev_Ak.T )) + Qk
#                    Kip[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = prev_diag
#                    
#                else:
#                    #curr_off_diag = np.dot( Ak, np.dot(prev_off_diag, prev_Ak.T) ) + np.dot(Ak, Qk)
#                    curr_off_diag = np.dot( Ak, prev_diag)
#                    curr_diag = np.dot( Ak, np.dot(prev_diag, Ak.T )) + Qk
#                
#                    Kip[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = curr_diag
#                    Kip[row_ind_start:row_ind_end, col_ind_start:col_ind_end] = curr_off_diag
#                    Kip[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = curr_off_diag.T
#                    
#                    prev_diag = curr_diag
#                
#                dAkd_k = np.empty( (grad_params_no, block_size, block_size) )
#                dCkd_k = np.empty( (grad_params_no, block_size, block_size) )
#                
#                dAk = AQcomp.dAk(k)
#                dQk = AQcomp.dQk(k)
#                
#                for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
#                    dAk_p = dAk[:,:,dd]
#                    dQk_p = dQk[:,:,dd]
#                    dQk_p = 0.5*(dQk_p + dQk_p.T)
#         
#                    sparse_dAit = Ait_derivatives[dd]
#                    sparse_dQi = Qi_derivatives[dd]
#                    
#                    tmp1 = -np.dot(Qk_inv, np.dot(dQk_p, Qk_inv))
#                    sparse_dAit[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = -dAk_p.T
#                    sparse_dQi[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = tmp1
#                    
#                    dAkd_k[dd,:,:] = np.dot( dAk_p.T, np.dot( Qk_inv, Ak) )
#                    dAkd_k[dd,:,:] += dAkd_k[dd,:,:].T
#                    
#                    tmp2 =  np.dot( Qk_inv, Ak) 
#                    dAkd_k[dd,:,:] += -np.dot(tmp2.T, np.dot(dQk_p, tmp2)) + d_Qk_inv_prev[dd]
#                    d_Qk_inv_prev[dd,:,:] = tmp1
#                    
#                    dCkd_k[dd,:,:] = -np.dot( Qk_inv, dAk_p) - np.dot(tmp1,Ak )
#                    
#                dAkd[k] = dAkd_k
#                dCkd[k] = dCkd_k
#                    
#            Qk_inv_prev = Qk_inv # required for derivative of determinant computations      
#                      
#        
#        Qi = Qi.asformat('csc')
#        Ait = Ait.asformat('csc')
#        #Bkd[k] = - np.dot( Ak.T, Qk_inv )
#        
#        Akd[x_points_num-1] = Qk_inv_prev # set the last element of the matrix        
#        if compute_derivatives:                 
#            dAkd[x_points_num-1] = d_Qk_inv_prev # set the last element of the matrix
#            
#            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
#                dP_inf_p = dP_inf[:,:,dd]            
#            
#                sparse_dQi = Qi_derivatives[dd]
#                sparse_dQi[0:block_size,0:block_size] = -np.dot(P_inf_inv, np.dot(dP_inf_p, P_inf_inv))
#                
#                sparse_dQi = sparse_dQi.asformat('csc')
#                sparse_dAit = Ait_derivatives[dd]
#                sparse_dAit = sparse_dAit.asformat('csc')                
#                
#                Ki_der = (sparse_dAit * Qi) * Ait.T # TODO: maybe this is block matrix
#                Ki_der += Ki_der.T
#                Ki_der += (Ait*sparse_dQi)*Ait.T                
#                
#                Ki_derivatives.append(Ki_der)
#                                
#         
#        GtY = G.T * sparse.csc_matrix(Y)
#        #GtY[H_nonzero_inds,0] = Y[0,0]* Hnn     # insert the first block
#        GtY = GtY.asformat('csc')  
#        #GtY = sparse.kron( Y, H.T, format='csc') # another way to compute
#        
#        
#        matrix_blocks = {}
#        matrix_blocks_derivatives = {}
#        matrix_blocks['Akd'] = Akd
#        matrix_blocks['Bkd'] = Bkd
#        if compute_derivatives:
#            matrix_blocks_derivatives['dAkd'] = dAkd
#            matrix_blocks_derivatives['dCkd'] = dCkd
#        
#        return Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, matrix_blocks_derivatives
#
#    @staticmethod
#    def marginal_ll(block_size, Y, Ait, Qi, GtY, G, GtG, H, g_noise_var, 
#                    compute_derivatives=False, dKi_vector=None, Kip=None,
#                    matrix_blocks=None, matrix_blocks_derivatives=None):
#        """
#        Function computes  marginal likelihood and its derivatives.        
#        
#        Inputs are mostly the necessary matrices obtained from the function "sparse_inverse_cov".
#        Input:        
#        -----------------------
#        compute_inv_main_diag: bool
#            Whether to compute intermidiate data for inversion of sparse
#            tridiagon precision. This is needed for further variance calculation.
#            For marginal likelihood and its gradient it is not required.
#        
#        Kip: sparse(Ai_size, Ai_size)
#            Block tridiagonal matrix. The tridiagonal blocks are the same as in K (inverse of Ki)
#            Required for ll derivarive computation.
#            TODO: note, because it is computed reqursively there may be a loss of accuracy!
#        """
#        measure_timings = True        
#        
#        
#        if measure_timings:
#            meas_times = {}
#            meas_times_desc = {}
#        
#            sparsity_meas = {}
#            sparsity_meas_descr = {} 
#        
#            meas_times_desc[1] = "Cov matrix multiplication"
#            meas_times_desc[2] = "Cholmod: analyze 1"
#            meas_times_desc[3] = "Cholmod: Cholesky in-place Ki"
#            meas_times_desc[4] = "Cholmod: Cholesky in-place KiN"
#            meas_times_desc[5] = "Derivative calculation part 1"
#            meas_times_desc[6] = "Derivative calculation part 2"
#            meas_times_desc[7] = "LML calculation main"            
#            
#            meas_times[7] = []
#            meas_times_desc[8] = "Der p2: data fit term"
#            
#            meas_times[8] = []
#            meas_times_desc[9] = "Der p2: determ deviratives (extra function)"
#            
#            meas_times[9] = []
#            meas_times_desc[10] = "Der noise: part 1"
#            
#        
#        HtH = np.dot(H.T, H)
#        if measure_timings: t1 = time.time()
#        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
#        Ki = 0.5*(Ki + Ki.T)
#        if measure_timings: meas_times[1] = time.time() - t1
#        g_noise_var = float(g_noise_var)
#        
#        KiN = Ki +  GtG /g_noise_var# Precision with a noise
#        if measure_timings: sparsity_meas[1] = Ki.getnnz()
#
#        if measure_timings: t1 = time.time()
#        analyzed_factor = cholmod.analyze(KiN) # perform only once this operation
#                                              # since this is expensive?  
#        if measure_timings: meas_times[2] = time.time() - t1        
#        
#        Ki_factor = analyzed_factor._clone()
#        KiN_factor = analyzed_factor._clone()
#        
#        if measure_timings: t1 = time.time()
#        # Ki_factor.cholesky_inplace(Ki) # There we error sometimes here, hence substitute to equivalent
#        Ki_factor.cholesky_inplace(Qi)
#        if measure_timings: meas_times[3] = time.time() - t1
#            
#        # Check 1 ->
##        Ko = np.linalg.inv(Ki.toarray())
##        K = G * Ko * G.T + np.eye(3) * g_noise_var
##        right_log_det = np.log( np.linalg.det(K) )
##        right_data_fit = np.dot(Y.T, np.dot( np.linalg.inv(K), Y ))
##        
##        p1_det = Ki_factor.logdet()
#        # Check 1 <-
#        
#        # Check 3 ->
##        dKi1 = dKi_vector[0]
##        dKi2 = dKi_vector[1]
##        
##        Ki_inv = Ki_factor.inv()
##        d1 = - G * Ki_inv * dKi1 * Ki_inv * G.T # dK_dTheta
##        d2 = - G * Ki_inv * dKi2 * Ki_inv * G.T
##        
##        K_inv = np.linalg.inv(K)
##        right_det_deriv1 = -0.5*np.trace( np.dot(K_inv, d1.toarray() )) 
##            
##        right_det_deriv2 = -0.5*np.trace( np.dot(K_inv, d2.toarray() ))
##        
##        tt1 = np.dot(K_inv,Y)
##        right_data_fit_deriv1 = 0.5*np.dot( tt1.T, np.dot( d1.toarray(), tt1) )
##        right_data_fit_deriv2 = 0.5*np.dot( tt1.T, np.dot( d2.toarray(), tt1) )
##        
##        right_der1 = right_det_deriv1 + right_data_fit_deriv1
##        right_der2 = right_det_deriv2 + right_data_fit_deriv2
#        # Check 3 <-
#        if measure_timings: t1 = time.time()
#        KiN_factor.cholesky_inplace(KiN, beta=0)        
#        if measure_timings: meas_times[4] = time.time() - t1
#            
#        data_num = Y.shape[0] # number of data points
#        deriv_number = len(dKi_vector) + 1 # plus Gaussian noise
#        d_marginal_ll = np.zeros((deriv_number,1))        
#        
#        if measure_timings: t1 = time.time()
#        if compute_derivatives:
#            #deriv_factor = analyzed_factor._clone()
#            for dd in range(0, deriv_number-1): #  the last parameter is Gaussian noise
#            
#                dKi = dKi_vector[dd]
##                if measure_timings: sparsity_meas[5].append(dKi.getnnz())
##                # First part of determinant
##                if measure_timings: t2 = time.time()
##                deriv_factor.cholesky_inplace(dKi, beta=0)
##                if measure_timings: meas_times[7].append(time.time() - t2)
##                    
##                (deriv_L, deriv_D) = deriv_factor.L_D()
##                deriv_L = deriv_L.tocsc()
##                deriv_D = deriv_D.tocsc()
##                
##                if measure_timings: t2 = time.time()
##                #L4 = Ki_factor.apply_P( deriv_factor.apply_Pt(deriv_L) )
##                L4=deriv_L
##                if measure_timings: meas_times[8].append(time.time() - t1)
##            
##                Ki_deriv_L.append(L4) # Check that there ara not need to do permutation
##                Ki_deriv_D.append(deriv_D)
#                
##                if measure_timings: t2 = time.time()
##                L5 = Ki_factor.solve_L(L4) # Same size as KiN
##                if measure_timings: meas_times[9].append(time.time() - t2)
##                if measure_timings: sparsity_meas[2].append(L5.getnnz())
##                
##                if measure_timings: t2 = time.time()
##                dd1 = (Ki_factor.solve_D(L5)*deriv_D).multiply(L5).sum() * 0.5
##                if measure_timings: meas_times[10].append(time.time() - t2)
#                    
#                #ss1 = m1.sum()
##               d_marginal_ll[dd,1] = -0.5*(ss) 
#            
#                # Another way ->
#                dd3 =  Kip.multiply(dKi.T).sum() * 0.5          
#                # Another way <-                
#                
#                
#                # Another way 2 ->
#                #ss = Ki_factor.solve_A(dKi)            
#                #dd2 = -0.5*( -ss.diagonal().sum() )
#                # Another way 2 <-
#                d_marginal_ll[dd,0] = dd3
#        if measure_timings: meas_times[5] = time.time() - t1        
#        # Check 2 ->        
#        #p2_det = Ki_factor.logdet()
#        #p3_det = (Y.size)*np.log( g_noise_var )
#        
#        #det2 = p2_det - p1_det + p3_det
#        # Check 2 <- 
#        
#        # Contribution from det{ K^{-1} }
#        if measure_timings: t1 = time.time()
#        mll_log_det = -Ki_factor.logdet()# ignoring 0.5 and - sign.
#        mll_log_det += KiN_factor.logdet() + (Y.size)*np.log(g_noise_var)
#        if np.isnan(mll_log_det):
#            raise ValueError("marginal ll: mll_log_det is None")
#        
#        KiNGtY = KiN_factor.solve_A(GtY)
#        
#        tmp1 = (GtY.T*KiNGtY).toarray()
#        
#        mll_data_fit_term = ( np.dot(Y.T,Y) /g_noise_var - tmp1/g_noise_var**2 ) # ignoring 0.5 and - sign.
#        
#        marginal_ll = -0.5 * ( mll_log_det + mll_data_fit_term + (Y.size)*log_2_pi)
#        if measure_timings: meas_times[7] = time.time() - t1
#            
#        if measure_timings: t1 = time.time()
#        d_deriv_true = []
#        if compute_derivatives: 
#            if measure_timings: t2 = time.time() # TODO maybe reimplement by stacking
#            for dd in range(0, deriv_number-1): #  the last parameter is Gaussian noise (ignore it)
#                dKi = dKi_vector[dd]
##                dL = Ki_deriv_L[dd]
##                dD = Ki_deriv_D[dd]
#                
#                # Another pard of determinant
##                ss = KiN_factor.solve_A( Ki_deriv_L[dd] )
##                ss = ss.power(2).sum()
##                d_marginal_ll[dd,1] += -0.5*(ss)
#                
##                if measure_timings: t2 = time.time()
##                L5 = KiN_factor.solve_L( dL)
##                if measure_timings: meas_times[11].append(time.time() - t2)
##                if measure_timings: sparsity_meas[3].append(L5.getnnz())                
##                
##                if measure_timings: t2 = time.time()
##                dd1 = -(KiN_factor.solve_D(L5)*dD).multiply(L5).sum() * 0.5
##                if measure_timings: meas_times[12].append(time.time() - t2)
##                
##                
##                # Another way ->
#                #ss = KiN_factor.solve_A(dKi)            
#                #dd2 = -0.5*( ss.diagonal().sum() )
##                d_deriv_true.append( dd2 )
#                # Another way <-
#                #d_marginal_ll[dd,0] += dd1            
#                
#                # 
#                
#                d_marginal_ll[dd,0] += -0.5* (KiNGtY.T*(dKi*KiNGtY)).toarray()/g_noise_var**2 # Data fit term
#            if measure_timings: meas_times[8].append(time.time() - t2)
#            
#            if measure_timings: t2 = time.time() # TODO maybe reimplement by stacking
#            #d_determ, det_for_test = sparse_inference.second_deriv_determinant( data_num, 
#            #            block_size, KiN, HtH, g_noise_var, dKi_vector, determ_derivatives)
#
## Use function deriv_determinant ->            
#            (d_determ, det_for_test, tmp_inv_data) = sparse_inference.deriv_determinant( data_num, \
#                                 block_size, HtH, g_noise_var, \
#                                 matrix_blocks, None, compute_derivatives=True, deriv_num=deriv_number, \
#                                 matrix_derivs=matrix_blocks_derivatives, compute_inv_main_diag=False)
## Use function deriv_determinant <-
#                                 
## Use function deriv_determinant2 ->                                 
##            (d_determ, det_for_test, tmp_inv_data) = sparse_inference.deriv_determinant2( data_num, block_size, HtH, g_noise_var, 
##                                 matrix_blocks, None, matrix_blocks_derivatives, None,
##                                 compute_inv_diag=False, add_noise_deriv=True)
## Use function deriv_determinant2 <-                                 
#            import pdb; pdb.set_trace()                     
#            d_marginal_ll += -0.5* d_determ[:,np.newaxis]
#            
#            if measure_timings: meas_times[9].append(time.time() - t2)
#                
#            # Derivative wrt noise
#            # Data term:
#            if measure_timings: t2 = time.time()
#            tmp2 = G*KiNGtY
#            d_marginal_ll[-1,0] += -0.5* ( -1.0/g_noise_var**2 * np.sum( np.power(Y,2)) + \
#                                            2.0/g_noise_var**3 * tmp1  - \
#                                            1.0/g_noise_var**4 * (tmp2.T * tmp2) )
#            if measure_timings: meas_times[10] = time.time() - t2
#                
##            # Detarminant terms:
##            if measure_timings: t2 = time.time()
##            ss = KiN_factor.solve_A(GtG)  # TODO work on that
##            if measure_timings: meas_times[15] = time.time() - t2
##            if measure_timings: sparsity_meas[4] = ss.getnnz()
#            
#            #d_marginal_ll[-1,0] += -0.5*( -ss.diagonal().sum()/g_noise_var**2 + Y.size/g_noise_var )
#            d_marginal_ll[-1,0] += -0.5*( Y.size/g_noise_var )
#                
#        if measure_timings: meas_times[6] = time.time() - t1
#        
#        
#        if measure_timings:
#            return marginal_ll, d_marginal_ll, meas_times, meas_times_desc, sparsity_meas, \
#                sparsity_meas_descr
#        else:
#            return marginal_ll, d_marginal_ll
#    
#    @staticmethod
#    def mean_var_calc(block_size, Y, Ait, Qi, GtY, G, GtG, H, g_noise_var, 
#                    matrix_blocks, which_observed=None, inv_precomputed=None):
#        """
#        Input:        
#        -----------------------
#        block_size: int
#            Size of the block
#            
#        Y: array(N,1)
#            1D-array.  For test points there are zeros in corresponding positions.
#            In the same positions where zeros are in which_observed.
#        
#        Ait, Qi, GtY, G, GtG, H: matrices
#            Matrix data
#        
#        g_noise_var: float
#            Noise variance
#            
#        matrix_blocks: dict
#            Data with matrices info past directly into deriv_determinant.
#        
#        which_observed: None, or array(N,1) or array(1,N) or array(N,) 
#            Array consisting of zeros and ones. Ones are in the position of
#            training points (observed), zeros are in the position of 
#            test points (not observed). If None, then all observed.
#            Affects whether we add or not a diagonal.
#            
#        inv_precomputed: 
#            What determinant computation function returns.
#        
#        Kip: sparce (not used anymore)
#            Block diagonal matrix. The diagonal blocks are the same as in K.
#            Required for ll derivarive computation.
#        """
#        g_noise_var = float(g_noise_var)
#        HtH = np.dot(H.T, H)
#        data_num = Y.shape[0]        
#        #import pdb; pdb.set_trace()
#        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
#        Ki = 0.5*(Ki + Ki.T)
#        
##        last_data_points = 0
##        diagg = np.ones((data_num,)); diagg[-last_data_points:] = 0
##        g_noise_var_to_det = diagg * g_noise_var; g_noise_var_to_det[-last_data_points:] = np.inf
##        GtG2 = sparse.kron( sparse.csc_matrix(np.diag(diagg)), sparse.csc_matrix( HtH ), format='csc')    
##        KiN = Ki +  GtG2 /g_noise_var# Precision with a noise
#        if which_observed is not None:
#            tmp1 = sparse.csc_matrix( G.T * sparse.csr_matrix( np.diag(np.squeeze(which_observed/g_noise_var)) ) )
#            GtG = tmp1 * G
#            
#            GtNY = tmp1*sparse.csc_matrix(Y)
#        
#        else:
#            GtG  = GtG/g_noise_var # Precision with a noise
#            
#            GtNY = GtY/g_noise_var
#            
#        KiN = Ki +  GtG
#        
#        analyzed_factor = cholmod.analyze(KiN) # perform only once this operation
#                                              # since this is expensive?  
#                
#        KiN_factor = analyzed_factor._clone()
#        KiN_factor.cholesky_inplace(KiN, beta=0)        
#        
#        KiNGtY = KiN_factor.solve_A(GtNY)
#        
#        if inv_precomputed is None:
#            _, _, inv_precomputed = sparse_inference.deriv_determinant( data_num, \
#                                 block_size, HtH, g_noise_var, \
#                                 matrix_blocks, which_observed, compute_derivatives=False, deriv_num=None, \
#                                 matrix_derivs=None, compute_inv_main_diag=True)
#        
#        #import pdb; pdb.set_trace()          
#        # compute extra matrix which required in covariance ->
#        Gt = sparse.csc_matrix( G.T )
#
##        # Get rid of off diagonal blocks ->
##        # mask to choose the block diagonal part
##        ones_mask = sparse.kron( sparse.eye( data_num, format='csc') ,sparse.csc_matrix( np.ones( (block_size, block_size) ) ), format='csc' )
##        tmp9 = ones_mask.multiply(Kip)
##        # Get rid of off diagonal blocks <-        
#        
#        #tmp12 = sparse.kron( sparse.csc_matrix( np.ones( (data_num,1) ) ), sparse.eye(block_size, format='csc'), format='csc')
#        tmp12 = sparse.csc_matrix( np.ones( (data_num,1) ) )
#        
#        #cov_calc_rhs = (tmp11*tmp12).toarray()
#        cov_calc_rhs = (Gt*tmp12).toarray()
#        # compute extra matrix which required in covariance <-
#        
##        # other (straightforward) result diag (test) ->
#        #Ki_factor = analyzed_factor._clone()        
#        #Ki_factor.cholesky_inplace(Ki, beta=0) 
#        
##        # diagonal part of the variance
##        diag_var = (G*tmp10*sparse.csc_matrix( np.ones( (data_num,1) ) )).toarray()
##        
##        om2 = sparse.kron( sparse.eye( data_num, format='csc') ,sparse.csc_matrix( np.ones( (block_size, 1) ) ), format='csc' )
##        K = Ki_factor.solve_A( sparse.eye( data_num*block_size, format='csc') )        
##        cc_rhs = GtG * (K * Gt / g_noise_var)
##                
##        tt1 = KiN_factor.solve_A(cc_rhs)
##        other_res_diag = np.diag((G*tt1).toarray()); other_res_diag.shape = (other_res_diag.shape[0],1)
##        var = diag_var - other_res_diag
##        # other (straightforward) result diag (test) <-
#        
#        # Compute mean
#        #mean_rhs = sparse.csc_matrix(Y/g_noise_var) - G*KiNGtY/g_noise_var # first term in this formula is incorrect. Need to account for inf. vars
#        #mean = (G*Ki_factor.solve_A( Gt*mean_rhs)).toarray()
#        
#        mean2 = (G*KiNGtY).toarray()
#        #import pdb; pdb.set_trace()   
#        #Compute variance
#        result_diag = sparse_inference.sparse_inv_rhs(data_num, block_size, 
#                        matrix_blocks, HtH/g_noise_var , H, inv_precomputed, cov_calc_rhs,
#                        which_observed) 
#        
#        #!!! HtH - H in sparse_inverse_cov and marginal_ll
#        # One extra input in sparse_inv_rhs
#                       
#        return mean2, result_diag
#    
#
#
#    @staticmethod
#    #@profile
#    def deriv_determinant2( points_num, block_size, HtH, g_noise_var, 
#                                 matrix_data, rhs_matrix_data, front_multiplier, which_observed=None,
#                                 compute_inv_diag=False, add_noise_deriv=False):
#        """
#        This function is a different implementation of determinant term 
#        (and its derivaives) in GP 
#        marginal likelihood. It uses the formula d(log_det)/d (Theta) = Trace[ K d(K)\d(Theta)]
#        Matrix K is assumed block tridiagonal, K^{-1} is sparse. Essentially what the function does:
#        it compues the diagonal of (K d(K)\d(Theta) ). Having a diagonal it computes a trace. 
#        
#        Right now the summetricity assumption is used, but in general the method
#        works for nonsymmetrix block tridiagonal matrices as well.        
#        
#        This function works for multiple rhs matrices so that all the dirivatives are computed at once!!!
#        
#        Notation for matrix is: K = A0, B1 0       
#                                    C1, A1, B2
#                                    0 , C2, A2
#        Input:
#        -----------------------------------
#        
#        points_num: int
#            Number blocks
#            
#        block_size: int
#            Size of the block            
#        
#        HtH: matrix (block_size, block_size)
#            Constant matrix added to the diagonal blocks. It is divieded by g_noise_var
#        
#        g_noise_var: float
#            U sed in connection with previous parameter for modify main diagonal blocks
#            
#        matrix_data: dictionary
#            It is supposed to contain 2 keys: 'Ak' and 'Bk'. Each of those
#            contain dictionary for main diagonal and upper-main diagonal.
#            They are accessible by indices matrix_data['Ak'][0:points_num],
#            matrix_data['Bk'][0:(points_num-1)]
#            
#        rhs_matrix_data: disctionary
#            The same format as in "matrix_data".
#            
#        front_multiplier: (block_size, k). k- any matrix
#            Matrix by which the whole expession K^{-1}D is multiplied in front.
#            Actually, small block of this matrix.
#        
#         which_observed: None, or array(N,1) or array(1,N) or array(N,) 
#            Array consisting of zeros and ones. Ones are in the position of
#            training points (observed), zeros are in the position of 
#            test points (not observed). If None, then all observed.
#            Affects whether we add or not a diagonal.
#            
#        add_noise_deriv: bool
#            Whether determinant noise derivative is computed. In this case, add one more derivative
#                wrt noise. dAkd have the derivative HtH wrt noise. dCkd have zero.
#        
#        deriv_num: int
#            Number of derivatives
#        
#        Output:
#        -------------------------------------
#        
#        d_determinant: array(deriv_num,1) or None       
#            Derivative of the determiant or None (if compute_derivatives == False)            
#        
#        determinant: float
#            Determiant
#        """
#        
#        Akd = matrix_data['Akd']
#        Bkd = matrix_data['Bkd']
#        
#        rhs_Akd = rhs_matrix_data['dAkd']
#        rhs_Ckd = rhs_matrix_data['dCkd']
#         # Convert input matrices to 
#        #HtH = HtH.toarray()
#        inversion_comp={}; inversion_comp['d_l'] = {}; inversion_comp['d_r'] = {}
#        inversion_comp['rhs_diag_d'] = {}; inversion_comp['rhs_diag_u'] = {}
#        
#        if which_observed is None:
#            which_observed = np.ones( points_num )
#        else:
#            which_observed = np.squeeze(which_observed )
#        
#        HtH_zero = np.zeros( HtH.shape )
#        extra_diag = lambda k: HtH if (which_observed[k] == 1) else HtH_zero
#        
#        prev_Lambda = Akd[0] + extra_diag(0)/g_noise_var #KiN[0:block_size, 0:block_size].toarray() #Akd[0] + HtH/g_noise_var
#        prev_Lambda_back = Akd[ (points_num-1) ] + extra_diag((points_num-1))/g_noise_var
#        
#        determinant = 0
#        d_determinant = None
#        deriv_num = rhs_Akd[0].shape[0]
#        if add_noise_deriv:
#            deriv_num += 1
#            
#        d_determinant = np.zeros( (deriv_num,) )
#            
#        rhs_Ck = np.zeros( (deriv_num, block_size, block_size ) )
#        rhs_Ak = np.zeros( (deriv_num, block_size, block_size ) )
#        rhs_Ck_back = np.zeros( (deriv_num, block_size, block_size ) )
#        rhs_Ak_back = np.zeros( (deriv_num, block_size, block_size ) )
#        
#        # In our notation the matrix consist of lower diagonal: C1, C1,...
#        # main diagonal: A0, A1, A2..., and upper diagonal: B1, B2,...        
#        # In this case the matrix is symetric.
#        
#        for i in range(0, points_num): # first point was for initialization
#        
#            (LL, cc) = la.cho_factor(prev_Lambda, lower = True)
#            (LL1, cc1) = la.cho_factor(prev_Lambda_back, lower = True)
#            
#    
#            inversion_comp['d_l'][i] = prev_Lambda 
#            inversion_comp['d_r'][points_num-i-1] = prev_Lambda_back
#                   
#            determinant += 2*np.sum( np.log(np.diag(LL) ) ) # new
#            
#            # HELP 2 ->            
#            # If we want to stack separate matrices in one dimansion
#            # e. g. A(n,m,m) -> A(m, m*n), we use:
#            # B = A.swapaxes(1,0).reshape(m,m*n)
#            
#            # If we want to transform back:
#            # A = B.reshape(m,n,m).swapaxes(1,0)           
#
#            if (i==points_num-1):
#                break
#                # Future points are not computed any more
#            
#            Bk = Bkd[i]# KiN[ind_start_lower:ind_end_lower, ind_start_higher:ind_end_higher].toarray() # Bkd[i]#
#            Ak = Akd[i+1]+ extra_diag(i+1)/g_noise_var# KiN[ind_start_higher:ind_end_higher, ind_start_higher:ind_end_higher].toarray() #Akd[i+1]+ HtH/g_noise_var#
#            
#            Bk_back = Bkd[points_num-i-2] 
#            Ak_back = Akd[points_num-i-2]+ extra_diag(points_num-i-2)/g_noise_var            
#            if add_noise_deriv:
#                rhs_Ck[0:-1,:,:] = rhs_Ckd[i];
#                rhs_Ak[0:-1,:,:] = rhs_Akd[i+1]; rhs_Ak[-1,:,:] = extra_diag(i+1)
#                
#                rhs_Ck_back[0:-1,:,:] = rhs_Ckd[points_num-i-2]
#                rhs_Ak_back[0:-1,:,:] = rhs_Akd[points_num-i-2]; rhs_Ak_back[-1,:,:] = extra_diag(points_num-i-2)
#            else:
#                rhs_Ck = rhs_Ckd[i]; 
#                rhs_Ak = rhs_Akd[i+1]
#                
#                rhs_Ck_back = rhs_Ckd[points_num-i-2]
#                rhs_Ak_back = rhs_Akd[points_num-i-2]
#
#            # Compute rhs_Ak_tmp part ->
#            tmp1 = np.transpose(rhs_Ck, (0,2,1) ) # first transpoce Ck to make it Bk
#            tmp1 = tmp1.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # solve a system
#            # Compute rhs_Ak_tmp part <-
#            
#            # Compute rhs_Ak_back_tmp part ->
#            tmp2 = rhs_Ck_back.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # solve a system 
#            # Compute rhs_Ak_back_tmp part <-
#            
#            # for doing every derivative simultaneously.
#            rhs_Ak_tmp = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num)
#            rhs_Ak_back_tmp = rhs_Ak_back.swapaxes(1,0).reshape(block_size,block_size*deriv_num)
#            
#            inversion_comp['rhs_diag_d'][i+1] = rhs_Ak_tmp - np.dot( Bk.T, la.cho_solve((LL, cc), tmp1 ) ) #tmp1
#            inversion_comp['rhs_diag_u'][points_num-i-2] = rhs_Ak_back_tmp - np.dot( Bk, la.cho_solve((LL1, cc1), tmp2 ) ) 
#            
#            if i == 0: # incert necessary matrices for the first and last points
#                rhs_Ak[0:-1,:,:] = rhs_Akd[0]; rhs_Ak[-1,:,:] = extra_diag(0)
#                inversion_comp['rhs_diag_d'][0] = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # forward transform
#                
#                rhs_Ak[0:-1,:,:] = rhs_Akd[points_num-1]; rhs_Ak[-1,:,:] = extra_diag(points_num-1)
#                inversion_comp['rhs_diag_u'][points_num-1] = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num)  # forward transform
#        
#            
#            prev_Lambda_inv_term = la.cho_solve((LL, cc), Bk) # new            
#            Lambda = Ak - np.dot(Bk.T, prev_Lambda_inv_term)
#            prev_Lambda = Lambda # For the next step
#        
#            prev_Lambda_inv_term1 = la.cho_solve((LL1, cc1), Bk_back.T) # new 
#            Lambda_back = Ak_back - np.dot( Bk_back, prev_Lambda_inv_term1)             
#            prev_Lambda_back = Lambda_back
#            del prev_Lambda_inv_term1, prev_Lambda_inv_term
#        
#        if front_multiplier is None:
#            new_block_size = block_size
#        else:
#            new_block_size = front_multiplier.shape[0]
#   
#        Akd = matrix_data['Akd']
#        d_l = inversion_comp['d_l']
#        d_r = inversion_comp['d_r']
#        d_d = inversion_comp['rhs_diag_d']
#        d_u = inversion_comp['rhs_diag_u']        
#        
#        #result_diag = np.empty( (points_num*new_block_size, d_2 ) )
#        #import pdb; pdb.set_trace()
#        inv_diag = {}
#        for i in range(0, points_num):
#            #start_ind = block_size*i
#            
#            #lft = np.tile(np.eye( block_size), (deriv_num,) ) - d_d[i] - d_u[i]
#            if add_noise_deriv:
#                #rhs_Ck[0:-1,:,:] = rhs_Ckd[i];
#                rhs_Ak[0:-1,:,:] = rhs_Akd[i]; rhs_Ak[-1,:,:] = extra_diag(i)
#            else:
#                rhs_Ak = rhs_Akd[i]
#            tmp1 = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num)
#            
#            lft =  -tmp1 + d_d[i] + d_u[i]            
#            
#            tmp = np.linalg.solve( -Akd[i] - extra_diag(i)/g_noise_var + d_l[i] + d_r[i], lft )
#            
#            tmp = tmp.reshape(block_size,deriv_num,block_size).swapaxes(1,0) # inverse transformation
#            # Temporarily block the return of diagonal because we have several diagonals
#            
#            d_determinant += np.trace(tmp, axis1 = 1, axis2=2)
#            inv_diag[i] = tmp[0, :, :]            
#            
#        if add_noise_deriv:
#            d_determinant[-1] = -d_determinant[-1] / (g_noise_var**2)
#                
#        return d_determinant, determinant, inversion_comp
#
#                                    
#    @staticmethod
#    #@profile
#    def deriv_determinant( points_num, block_size, HtH, g_noise_var, 
#                                 matrix_data, which_observed=None, compute_derivatives=False, deriv_num=None, 
#                                 matrix_derivs=None, compute_inv_main_diag=False):
#        """
#        This function computes the log_detarminant,its derivatives: d_log_determinant,
#        and 3 diagonals (not implemented) of the inverse of SYMMETRIC BLOCK TRIDIAGONAL MATRIX.
#        
#        It uses the method of differentiating the recursive formula for determinant.
#        
#        Right now the summetricity assumption is used, but in general the method
#        works for nonsymmetrix block tridiagonal matrices as well.        
#        
#        Notation for matrix is: K = A0, B1 0       
#                                    C1, A1, B2
#                                    0 , C2, A2
#        Input:
#        -----------------------------------
#        
#        points_num: int
#            Number blocks
#            
#        block_size: int
#            Size of the block            
#        
#        HtH: matrix (block_size, block_size)
#            Constant matrix added to the diagonal blocks. It is divieded by g_noise_var
#        
#        g_noise_var: float
#            U sed in connection with previous parameter for modify main diagonal blocks
#            
#        matrix_data: dictionary
#            It is supposed to contain 2 keys: 'Ak' and 'Bk'. Each of those
#            contain dictionary for main diagonal and upper-main diagonal.
#            They are accessible by indices matrix_data['Ak'][0:points_num],
#            matrix_data['Bk'][0:(points_num-1)]
#            
#        which_observed: None, or array(N,1) or array(1,N) or array(N,) 
#            Array consisting of zeros and ones. Ones are in the position of
#            training points (observed), zeros are in the position of 
#            test points (not observed). If None, then all observed.
#            Affects whether we add or not a diagonal.
#            
#        compute_derivatives: bool
#            Whether to compute determinant derivatives
#        
#        deriv_num: int
#            Number of derivatives
#            
#        matrix_derivs: dictionary
#            Derivatives of matrix blocks. With keys 'dAkd', 'dCkd'. Indices are
#            similar to matrix_data parameter.
#        
#        compute_inv_main_diag: bool
#            Whether to compute some intermidiate numbers do that later it is
#            faster to compute the block diagonal of inverse of the matrix K.
#            
#        
#        
#        Output:
#        -------------------------------------
#        
#        d_determinant: array(deriv_num,1) or None       
#            Derivative of the determiant or None (if compute_derivatives == False)            
#        
#        determinant: float
#            Determiant
#            
#        inversion_comp: dict{} 
#            Dict with 2 keys 'd_l' and 'd_r'. There stored the information for computing
#            inverse of tridiagonal matrix. If 'compute_inv_main_diag' the None.
#        """
#        
#        Akd = matrix_data['Akd']
#        Bkd = matrix_data['Bkd']
#         # Convert input matrices to 
#        #HtH = HtH.toarray()
#        if isinstance(g_noise_var, np.ndarray):
#            noise_vector = True
#            noise_var = g_noise_var[0]
#        else:
#            noise_vector = False
#            noise_var = g_noise_var
#        
#        if which_observed is None:
#            which_observed = np.ones( points_num )
#        else:
#            which_observed = np.squeeze(which_observed )
#        
#        HtH_zero = np.zeros( HtH.shape )
#        extra_diag = lambda k: HtH if (which_observed[k] == 1) else HtH_zero
#        
#        prev_Lambda = Akd[0] + extra_diag(0)/noise_var #KiN[0:block_size, 0:block_size].toarray() #Akd[0] + HtH/g_noise_var
#        
#        determinant = 0
#        d_determinant = None
#        if compute_derivatives:
#            d_determinant = np.zeros( (deriv_num,) )
#            
#            dAkd = matrix_derivs['dAkd']
#            dCkd = matrix_derivs['dCkd']
#            
#            dCk = np.zeros( (deriv_num, block_size, block_size ) )
#            dAk = np.zeros( (deriv_num, block_size, block_size ) )
#            prev_d_Lambda = np.empty( (deriv_num, block_size, block_size ) ) 
#            #for j in range(0,deriv_num-1):
#            #    prev_d_Lambda[j, :, :] = dKi_vector[j][0:block_size, 0:block_size].toarray()
#            prev_d_Lambda[0:-1,:,:] = dAkd[0] # new
#            prev_d_Lambda[-1, :, :] = extra_diag(0) # (-HtH/g_noise_var**2) # HtH # new
#        
#        inversion_comp = None
#        if compute_inv_main_diag:
#                
#            inversion_comp={}; inversion_comp['d_l'] = {}; inversion_comp['d_r'] = {}
#            prev_Lambda_back = Akd[ (points_num-1) ] + extra_diag((points_num-1))/noise_var
#        
#        # In our notation the matrix consist of lower diagonal: C1, C1,...
#        # main diagonal: A0, A1, A2..., and upper diagonal: B1, B2,...        
#        # In this case the matrix is symetric.
#        
#        for i in range(0, points_num): # first point was for initialization
#            #import pdb; pdb.set_trace()
#            (LL, cc) = la.cho_factor(prev_Lambda, lower = True)
#            #KiN_ldet += np.log( la.det( prev_Lambda) ) # old         
#            determinant += 2*np.sum( np.log(np.diag(LL) ) ) # new
#                
#            if compute_derivatives:
#                # HELP 2 ->            
#                # If we want to stack separate matrices in one dimansion
#                # e. g. A(n,m,m) -> A(m, m*n), we use:
#                # B = A.swapaxes(1,0).reshape(m,m*n)
#                
#                # If we want to transform back:
#                # A = B.reshape(m,n,m).swapaxes(1,0)           
#                
#                # HELP 2 <-
#                tmp = prev_d_Lambda.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # solve a system 
#                # for doing every derivative simultaneously.
#                
#                tmp = la.cho_solve((LL, cc), tmp )  # new          
#                # tmp = la.solve(prev_Lambda, tmp) # old
#                tmp = tmp.reshape(block_size,deriv_num,block_size).swapaxes(1,0) # inverse transformation
#                
#                d_determinant += np.trace( tmp, axis1 = 1, axis2 = 2)
#            
#            if compute_inv_main_diag:
#                (LL1, cc1) = la.cho_factor(prev_Lambda_back, lower = True)
#                
#                inversion_comp['d_l'][i] = prev_Lambda 
#                inversion_comp['d_r'][points_num-i-1] = prev_Lambda_back
#                
#            #print(i)
#            if (i==points_num-1):
#                break
#                # Future points are not computed any more
#            
##            ind_start_higher = (i+1)*block_size
##            ind_end_higher = ind_start_higher + block_size
##            ind_start_lower = i*block_size            
##            ind_end_lower = ind_start_lower + block_size
#            if noise_vector:
#                noise_var = g_noise_var[i+1]
#                
#            Bk = Bkd[i]# KiN[ind_start_lower:ind_end_lower, ind_start_higher:ind_end_higher].toarray() # Bkd[i]#
#            Ak = Akd[i+1]+ extra_diag(i+1)/noise_var# KiN[ind_start_higher:ind_end_higher, ind_start_higher:ind_end_higher].toarray() #Akd[i+1]+ HtH/g_noise_var#
#            
#            #for j in range(0,deriv_num-1):
#                
#                #dCk[j, :, :] = dKi_vector[j][ind_start_higher:ind_end_higher, ind_start_lower:ind_end_lower].toarray()
#                #dAk[j, :, :] = dKi_vector[j][ind_start_higher:ind_end_higher, ind_start_higher:ind_end_higher].toarray()
#            
#            prev_Lambda_inv_term = la.cho_solve((LL, cc), Bk) # new            
#            Lambda = Ak - np.dot(Bk.T, prev_Lambda_inv_term)
#            prev_Lambda = Lambda # For the next step
#            
#            if compute_inv_main_diag:
#                Bk = Bkd[points_num-i-2] 
#                Ak = Akd[points_num-i-2]+ extra_diag(points_num-i-2)/noise_var
#            
#                prev_Lambda_inv_term1 = la.cho_solve((LL1, cc1), Bk.T) # new 
#            
#                Lambda_back = Ak - np.dot( Bk, prev_Lambda_inv_term1)             
#            
#                prev_Lambda_back = Lambda_back
#                del prev_Lambda_inv_term1
#                
#            if compute_derivatives:
#                dCk[0:-1, :, :] = dCkd[i] # new
#                dAk[0:-1, :, :] = dAkd[i+1] # new
#                dAk[-1, :, :] = extra_diag(i+1) # -HtH/g_noise_var**2 # new # HtH
#            
#                # HELP 1 ->
#                # If we have 3d-array A(n, m, m) and want to multiply by B(m, m) n times:
#                # e.g ( A[0] * B, A[1] * B,  ...) then we do:
#                # C = np.dot(A,B)
#                
#                # The same but with B.T:
#                # C = np.dot(A,B.T)
#                
#                # If we want to compute: B(m, m)* A(n ,m ,m) we do:
#                # C = np.transpose(np.dot(np.transpose(A,(0,2,1)),B.T),(0,2,1))
#                
#                # the same but: B.T * A
#                # C = np.transpose(np.dot(np.transpose(A,(0,2,1)),B),(0,2,1))            
#                
#                # ! Everywhere insted of np.dot we can use np.multiply !
#                # HELP 1 <-
#                d1 = np.dot( dCk, prev_Lambda_inv_term)
#                d2 = np.dot( prev_d_Lambda, prev_Lambda_inv_term)
#                d2 = np.transpose(np.dot(np.transpose(d2,(0,2,1)),prev_Lambda_inv_term),(0,2,1))
#                
#                d_Lambda = dAk - d1 - np.transpose(d1, (0,2,1)) + d2
#                prev_d_Lambda = d_Lambda # For the next step
#        
#        if compute_derivatives:
#            if noise_vector:
#                raise NotImplemented
#            d_determinant[-1] = -d_determinant[-1] / (g_noise_var**2)
#                
#        return d_determinant, determinant, inversion_comp
#    
#    @staticmethod    
#    def sparse_inv_rhs(points_num, block_size, matrix_data, 
#                       extra_matrix_block_diag, front_multiplier, 
#                       inversion_comp, rhs, which_observed=None):
#        """
#        Function computes diagonal blocks of the (inverse tri-diag times matrix)
#        given some precalculated data in the function 'deriv_determinant' and
#        passed in 'inversion_comp'.
#        
#        Inputs:
#        -----------------------
#        points_num:
#        
#        block_size:
#        
#        matrix_data: dictionary
#            It is supposed to contain 2 keys: 'Ak' and 'Bk'. Each of those
#            contain dictionary for main diagonal and upper-main diagonal.
#            They are accessible by indices matrix_data['Ak'][0:points_num],
#            matrix_data['Bk'][0:(points_num-1)]
#            
#        extra_matrix_block_diag: matrix(block_size, block_size)
#            On each iteration this matrix is added to the diagonal block.
#            It is typically HtH/g_noise_var.
#            
#        front_multiplier: matrix(*, block_size)
#            On each iteration this matrix is multiplied by the result of the inversion
#            
#        rhs: matrix(block_size, block_size) or matrix(block_size*points_num, block_size)
#            If it is a larger matrix then blocks are in given in a column.
#            If it smaller matrix then it is assumed that all blocks are the same and only one is given.
#        
#        which_observed: None, or array(N,1) or array(1,N) or array(N,) 
#            Array consisting of zeros and ones. Ones are in the position of
#            training points (observed), zeros are in the position of 
#            test points (not observed). If None, then all observed.
#            Affects whether we add or not a diagonal.
#            
#        Output:
#        -----------------------
#        """
#        (d_1, d_2) = rhs.shape
#        
#        if (d_1 == block_size):
#            rhs_small=True
#        elif (d_1 == block_size*points_num):
#            rhs_small=False
#        else:
#            raise ValueError("sparse_inv_rhs: Incorrect rhs.")
#        
#        if which_observed is None:
#            which_observed = np.ones( points_num )
#        else:
#            which_observed = np.squeeze(which_observed )
#        
#        HtH_zero = np.zeros( extra_matrix_block_diag.shape )
#        extra_diag = lambda k: extra_matrix_block_diag if (which_observed[k] == 1) else HtH_zero
#   
#        if front_multiplier is None:
#            new_block_size = block_size
#        else:
#            new_block_size = front_multiplier.shape[0]
#   
#        Akd = matrix_data['Akd']
#        d_l = inversion_comp['d_l']
#        d_r = inversion_comp['d_r']
#        
#        result_diag = np.empty( (points_num*new_block_size, d_2 ) )
#        for i in range(0, points_num):
#            start_ind = block_size*i
#            if rhs_small:
#                lft = rhs
#            else:
#                lft = rhs[start_ind:start_ind+block_size, :]
#
#            tmp = np.linalg.solve( -Akd[i] - extra_diag(i) + d_l[i] + d_r[i], lft )
#            
#            start_ind2 = new_block_size*i
#            if front_multiplier is None:
#                result_diag[start_ind2:(start_ind2+new_block_size),:] = tmp
#            else:
#                result_diag[start_ind2:(start_ind2+new_block_size),:] = np.dot(front_multiplier, tmp)
#            
#        return result_diag
#    
        
class btd_inference(object):
    

    class AQcompute_batch_Python(object):
        """
        Class for calculating matrices A, Q, dA, dQ of the discrete Kalman Filter
        from the matrices F, L, Qc, P_ing, dF, dQc, dP_inf of the continuos state
        equation. dt - time steps.


        It computes matrices for all time steps. This object is used when
        there are not so many (controlled by internal variable)
        different time steps and storing all the matrices do not take too much memory.
        """
        def __init__(self, F,L,Qc,dt,compute_derivatives=False, grad_params_no=None, P_inf=None, dP_inf=None, dF = None, dQc=None):
            """
            Constructor. All necessary parameters are passed here and stored
            in the opject.

            Input:
            -------------------
                F, L, Qc, P_inf : matrices
                    Parameters of corresponding continuous state model
                dt: array
                    All time steps
                compute_derivatives: bool
                    Whether to calculate derivatives

                dP_inf, dF, dQc: 3D array
                    Derivatives if they are required

            Output:
            -------------------
            Nothing
            """
            #import pdb; pdb.set_trace()
            As, Qs, reconstruct_indices, dAs, dQs = ssm.ContDescrStateSpace.lti_sde_to_descrete(F,
                        L,Qc,dt,compute_derivatives,
                        grad_params_no=grad_params_no, P_inf=P_inf, dP_inf=dP_inf, dF=dF, dQc=dQc)

            self.As = As
            self.Qs = Qs
            self.dAs = dAs
            self.dQs = dQs
            self.reconstruct_indices = reconstruct_indices
            self.total_size_of_data = self.As.nbytes + self.Qs.nbytes +\
                            (self.dAs.nbytes if (self.dAs is not None) else 0) +\
                            (self.dQs.nbytes if (self.dQs is not None) else 0) +\
                            (self.reconstruct_indices.nbytes if (self.reconstruct_indices is not None) else 0)

            self.Q_svd_dict = {}
            self.Q_inverse_dict = {}
            
            self.last_k = None
             # !!!Print statistics! Which object is created
            # !!!Print statistics! Print sizes of matrices

        def Ak(self,k,m,P):
            self.last_k = k
            return self.As[:,:, self.reconstruct_indices[k]]

        def Qk(self,k):
            self.last_k = k
            return self.Qs[:,:, self.reconstruct_indices[k]]

        def dAk(self,k):
            self.last_k = k
            return self.dAs[:,:, :, self.reconstruct_indices[k]]

        def dQk(self,k):
            self.last_k = k
            return self.dQs[:,:, :, self.reconstruct_indices[k]]

        def Q_inverse(self, k, p_largest_cond_num, p_regularization_type):
            """
            Function inverts Q matrix and regularizes the inverse.
            Regularization is useful when original matrix is badly conditioned.
            Function is currently used only in SparseGP code.
            
            Inputs:
            ------------------------------
            k: int
            Iteration number.
            
            p_largest_cond_num: float
            Largest condition value for the inverted matrix. If cond. number is smaller than that
            no regularization happen.
            
            regularization_type: 1 or 2
            Regularization type.
            
            regularization_type: int (1 or 2)
            
                type 1: 1/(S[k] + regularizer) regularizer is computed
                type 2: S[k]/(S^2[k] + regularizer) regularizer is computed
            """
            #import pdb; pdb.set_trace()
            
            matrix_index = self.reconstruct_indices[k]
            if matrix_index in self.Q_inverse_dict:
                new_S = None
                Q_inverse_r = self.Q_inverse_dict[matrix_index]
            else:
                
                if matrix_index in self.Q_svd_dict:
                    (U, S, Vh) = self.Q_svd_dict[matrix_index]
                else:
                    (U, S, Vh) = sp.linalg.svd( self.Qs[:,:, matrix_index],
                                        full_matrices=False, compute_uv=True,
                                        overwrite_a=False, check_finite=False)
                    self.Q_svd_dict[matrix_index] = (U,S,Vh)
                #if k ==0:
                #    import pdb; pdb.set_trace()
                    
                Q_inverse_r, new_S,_ = btd_inference.psd_matrix_inverse(k, 0.5*(self.Qs[:,:, matrix_index] + self.Qs[:,:, matrix_index].T), U,S, p_largest_cond_num, p_regularization_type)
                self.Q_inverse_dict[matrix_index] = Q_inverse_r

            return Q_inverse_r, new_S
            
    @staticmethod
    def psd_matrix_inverse(k,Q, U=None,S=None, p_largest_cond_num=None, regularization_type=2):
        """
        Function inverts positive definite matrix and regularizes the inverse.
        Regularization is useful when original matrix is badly conditioned.
        Function is currently used only in SparseGP code.
        
        Inputs:
        ------------------------------
        k: int
        Iteration umber. Used for information only. Value -1 corresponds to P_inf_inv.
        
        Q: matrix
        To be inverted
        
        U,S: matrix. vector
        SVD components of Q
        
        p_largest_cond_num: float
        Largest condition value for the inverted matrix. If cond. number is smaller than that
        no regularization happen.
        
        regularization_type: 1 or 2 or 3
        Regularization type.
        """
        #import pdb; pdb.set_trace()
    #    if (k == 0) or (k == -1): # -1 - P_inf_inv computation
    #        import pdb; pdb.set_trace()
        
        if p_largest_cond_num  is None:
            raise ValueError("psd_matrix_inverse: None p_largest_cond_num")
            
        if U is None or S is None:
            (U, S, Vh) = sp.linalg.svd( Q, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
        if S[0] < (1e-4):
            #import pdb; pdb.set_trace()
            warnings.warn("""state_space_main psd_matrix_inverse: largest singular value is too small {0:e}.
                condition number is {1:e}.
                """.format(S[0], S[0]/S[-1]))
            #S = S + (1e-4 - S[0]) # make the S[0] at least 1e-4
            
        current_conditional_number = S[0]/S[-1]
        if (current_conditional_number > p_largest_cond_num):
            if (regularization_type == 1):
                regularizer = S[0] / p_largest_cond_num
                # the second computation of SVD is done to compute more precisely singular
                # vectors of small singular values, since small singular values become large.
                # It is not very clear how this step is useful but test is here.
                Q_r  = Q + regularizer*np.eye(Q.shape[0])
                (U, S, Vh) = sp.linalg.svd( Q_r, 
                                            full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
                
                new_S = 1.0/S
                Q_inverse_r = np.dot( U * new_S , U.T ) # Assume Q_inv is positive definite    
                # In this case, RBF kernel we get complx eigenvalues. Probably
                # for small eigenvalue corresponding eigenvectors are not very orthogonal.
                ##########Q_inverse = np.dot( Vh.T * ( 1.0/(S + regularizer)) , U.T )
            elif (regularization_type == 2):
                 #if k == 0:
                 #    import pdb; pdb.set_trace()
                 if k==0:
                     warnings.warn("""state_space_main psd_matrix_inverse: Using regularization type 2. 
                     Iteration 0. This is either P_inf or first dt. Old condition number is {0:e}.
                     New one is {1:e}.
                    """.format(current_conditional_number, p_largest_cond_num))
                    
                 new_border_value = np.sqrt(current_conditional_number)/2 
                 if p_largest_cond_num >= new_border_value: # this type of regularization works
                    regularizer = ( S[0] / p_largest_cond_num / 2.0 )**2
                    
                    new_S = ( S/(S**2 + regularizer))
                    Q_inverse_r = np.dot( U * new_S, U.T ) # Assume Q_inv is positive definite
                    Q_r = np.dot( U * 1.0/new_S , U.T ) # Assume Q_inv is positive definite
                 else:
                    
                    better_curr_cond_num = new_border_value 
                    warnings.warn("""state_space_main psd_matrix_inverse: reg_type = 2 can't be done completely.
                        Current conditionakl number {0:e} is reduced to {1:e} by reg_type = 1""".format(current_conditional_number, better_curr_cond_num))
                    
                    regularizer = S[0]/ (2* p_largest_cond_num)**2  - S[-1]
                    # the second computation of SVD is done to compute more precisely singular
                    # vectors of small singular values, since small singular values become large.
                    # It is not very clear how this step is useful but test is here.
                    (U, S, Vh) = sp.linalg.svd( Q + regularizer*np.eye(Q.shape[0]), 
                                                full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
                                                
                    regularizer = ( S[0] / p_largest_cond_num / 2.0 )**2 *100
                    
                    new_S = ( S/(S**2 + regularizer))
                    Q_inverse_r = np.dot( U * new_S , U.T ) # Assume Q_inv is positive definite
                    Q_r = np.dot( U * 1.0/new_S , U.T ) # Assume Q_inv is positive definite
                    
            elif (regularization_type == 3): # NUllify the inverse of very small singular values
                assert False, "Reg type 3 not supported!"
                cutoff = S[0] / p_largest_cond_num                
                
                new_S = 1/S
                new_S[ S < cutoff ] = 0.0
                
                Q_inverse_r = np.dot( U * (new_S) , U.T )
                
            else:
                raise ValueError("AQcompute_batch_Python:Q_inverse: Invalid regularization type")
            
            assert np.sqrt(regularizer)*100 < S[0], "regularizer is not << S[0]"
         
            #assert np.sqrt(regularizer) > 100*S[-1], "regularizer is not >> S[-1]"
             
        else:
            
            new_S = 1.0/S
            Q_inverse_r = np.dot( U * 1.0/S , U.T ) # Assume Q_inv is positive definite
            Q_r = Q
        # When checking conditional number 2 times difference is ok.
        Q_inverse_r = 0.5*(Q_inverse_r + Q_inverse_r.T)

        return Q_inverse_r, new_S, Q_r
    
    @staticmethod 
    def block_tridiag_solver(N,K, A_matr, C_matr, rhs_D_matr, rhs_diag=False, inversion_comp=None, rhs_C_matr=None,
                             comp_matrix_determinant=False, comp_solution=False, comp_solution_trace=False,
                             rhs_block_num=None, rhs_block_width = None, front_multiplier=None):
        """
        This function solves all required block-tridiagonal subproblem.
        Now the problem is assumed to be symmetrix, hence only one
        SUBdiagnal (Cd)s are given
        
        Input:
        -----------------------
        N: int
            Number of data points
        K: int
            Block size
        
        A_matr: matrix [N*K, K]
            Block row matrix representing the main diagonal.        
        
        C_matr: matrix [(N-1)*K, K]
            Block row matrix representing subdiagonal.
            
        rhs_D_matr: matrix [N*K, rhs_block_num*rhs_block_width]            
            Right-hand-side or diagonal blocks of the RHS.
            
        rhs_diag: bool
            If False - rhs represent a vector (matrix) RHS, if true
            it represent a block-diagoanal of the inverse
            
        inversion_comp: dictionary or None
            If the same matrix (which is inverted) were procesed already then
            this dictionary can be use the second time. This is possible only
            for BTD_rhs. 
            
        rhs_C_matr: matrix [(N-1)*K, rhs_block_num*rhs_block_width]
            Subdiagonal of rhs.
            
        comp_matrix_determinant: bool
            Whether to compute original matrix determinant
        
        comp_solution: bool
            Whether to compute solution
            
        comp_solution_trace: bool 
            Only trace of the diagonal solution is required
            
        rhs_block_num: int
            It says how many BTD systems we are solving sumiltaneously.
            If more then one then the blocks in rhs_D_matr and rhs_C_matr 
            are of size [K, rhs_block_num*K]
            
        rhs_block_width: int
            For the case with multiple RHS says what is the width of each block
    
        front_multiplier: matrix[any, K]
            When the BTD matrix is inverted, the might be constant front multiplier which
            multiplies each part of the solution.
            
        Output:
        -----------------------
        r1:
            abs( LogDet[A] )
            
        r2: array
            Solution. For regular system just solution. For BTD the diagonal
            of the solution.
            
        r3: float
            For BTD system computes Trace[Solution]
            
        comp_inv: dict
            values obtained on the first (down iteration) which can be reused for
            different right hand sides (only for BTD rhs).
        """

        Ak = lambda i: A_matr[i*K:(i+1)*K, 0:K] # Dd_matr indexing from 0 to N-1
        Ck = lambda i: C_matr[i*K:(i+1)*K, 0:K] # Cd_matr indexing from 0 to N-2
        
        rhs_Dk = lambda i: rhs_D_matr[i*K:(i+1)*K, :] # Dd_matr indexing from 0 to N-1
        if rhs_C_matr is not None:
            rhs_Ck = lambda i: rhs_C_matr[i*K:(i+1)*K, :] # Cd_matr indexing from 0 to N-2
        
#        rhs_D = lambda i: rhs_D_matr[i*K:(i+1)*K, :]
#        if rhs_C_matr is not None:
#            rhs_C = lambda i: rhs_C_matr[i*K:(i+1)*K, :]
        
        if inversion_comp is None:
            inversion_comp={}; inversion_comp['forward_LU'] = {}; inversion_comp['forward_piv'] = {}
            inversion_comp['U'] = {}
            forward_comp = True
        else:
            forward_comp = False
        
        if forward_comp == False:
            assert comp_matrix_determinant == False, "Precomputed inversion and determinant coputations are incompatible"
            assert rhs_diag == True, "Precomputed inversion and not BTD_rhs are incompatible"
            
        if comp_matrix_determinant:
            logdetA = np.empty((N,))
            
        if comp_solution or comp_solution_trace:
            if not rhs_diag:
                solution = rhs_D_matr.copy() # Here the modification of rhs or rhs_diag is stored along the computations
            else:
                solution = np.empty( rhs_D_matr.shape )
            
            if rhs_C_matr is not None:
                rhs_diag_sub = True
            else:
                rhs_diag_sub = False
        
        if rhs_diag:
            if rhs_block_num is None:
                rhs_block_num = 1 # solving multiple RHS systems simultaneously.
            
            if rhs_block_num == 1:
                if rhs_block_width is None:
                    rhs_block_width = K
                    
            if rhs_C_matr is not None:
                assert (rhs_D_matr.shape[1] == rhs_block_width*rhs_block_num), "rhs_block_num is wrong 1"
                assert (K == rhs_block_width), "RHS non-square blocks suported only when rhs_C_matr is zero (no sub diagonals)"
                assert (rhs_C_matr.shape[1] == rhs_block_width*rhs_block_num), "rhs_block_num is wrong 2"
            
        if comp_solution_trace:
            assert rhs_diag, "Solution trace is aplicable only for BTD inversion."
            assert front_multiplier is None, "Solution trace and Front Multiplier are incompatible."
            assert rhs_block_width == K, "Trace is computable ony for square blocks in RHS."
            
            solution_trace = np.empty( (N, rhs_block_num) )
        
        use_front_multiplier = False
        if front_multiplier is not None:
            assert rhs_diag, "Front Multiplier is aplicable only for BTD inversion."
            front_mult_solution = np.empty( (front_multiplier.shape[0]*N, rhs_D_matr.shape[1]) )
            
            use_front_multiplier = True
        # In our notation the matrix consist of lower diagonal: C1, C1,...
        # main diagonal: A0, A1, A2..., and upper diagonal: B1, B2,...        
        # In this case the matrix is symetric.
            
        #test_inv = np.zeros((N*K, N*K))
        
        #import pdb; pdb.set_trace()
        # Forward pass: compute logdetA, inversion_comp, solution for regular inversion.
        # If we do BTD_rhs inversion this step can be skiped if it is done before.
        if forward_comp:
            prev_Lambda = Ak(0);
            for i in range(0, N): # first point was for initialization
            
                (LU, piv) = la.lu_factor(prev_Lambda)
    
                inversion_comp['forward_LU'][i] = LU 
                inversion_comp['forward_piv'][i] = piv
                # Actually this is neded for rhs_diag diag computations only:             
                inversion_comp['U'][i] = la.lu_solve((LU, piv),  Ck(i).T )
                
                if comp_matrix_determinant:
                    ddd = np.diag(LU)
                    logdetA[i] = np.sum( np.log(np.abs(ddd) ) ) # new
                    
                if (i==N-1):
                    break
                    # Future points are not computed any more
                
                if comp_solution or comp_solution_trace:
                    if not rhs_diag: # regular system
                        solution[(i+1)*K:(i+2)*K, : ] = solution[(i+1)*K:(i+2)*K, : ] - np.dot( Ck(i), la.lu_solve((LU, piv),  solution[(i+0)*K:(i+1)*K,:] ) )
                    else: # inverse of BTD matrix
                        # Nothing to do here, Only inversion_comp['U'][i] is relevant for this problem
                        pass
                              
                Lambda = Ak(i+1) - np.dot(Ck(i), la.lu_solve((LU, piv), Ck(i).T)) # New Lambda
                prev_Lambda = Lambda # For the next step
        
        #import pdb; pdb.set_trace()
        if comp_solution or comp_solution_trace:
            # Backward pass
            #  For BTD solution matrix A is notded here anymore. For regular system Ck is still used.
            for i in range(N-1, -1,-1): # first point was for initialization
                LU = inversion_comp['forward_LU'][i] 
                piv = inversion_comp['forward_piv'][i]
                
                # do the backward elimination step
                tmp_sol = None
                if i != 0:
                    if not rhs_diag: # regular system. Correcting the previous step
                        tmp_sol = la.lu_solve((LU, piv),  solution[(i+0)*K:(i+1)*K, : ] )
                        solution[(i-1)*K:(i)*K, : ] = solution[(i-1)*K:(i)*K, : ] - np.dot( Ck(i-1).T, tmp_sol )
                        
                    else: # inverse of BTD matrix
                        # Nothing to do here
                        pass
                    
                # computing solution
                if not rhs_diag: # regular system
                    if tmp_sol is not None: 
                        solution[(i+0)*K:(i+1)*K, : ] = tmp_sol
                    else:
                        solution[(i+0)*K:(i+1)*K, : ] = la.lu_solve((LU, piv),  solution[(i+0)*K:(i+1)*K, : ] )
                
                else: # inverse of BTD matrix
                    U = inversion_comp['U'][i]
                    if i == N-1:
                        #test_inv[ (i+0)*K:(i+1)*K, (i+0)*K:(i+1)*K] =  la.lu_solve((LU, piv),  np.eye(K) )   
                        solution[(i+0)*K:(i+1)*K, : ] = la.lu_solve((LU, piv), rhs_Dk(i) )
                        prev_Uup = np.zeros((K,K))
                    else:
                        prev_LU = inversion_comp['forward_LU'][i+1] 
                        prev_piv = inversion_comp['forward_piv'][i+1]
                        
                        up = - la.lu_solve((prev_LU, prev_piv),  U.T ).T + np.dot(U, prev_Uup)
                        #low = -np.dot( test_inv[ (i+1)*K:(i+2)*K, (i+1)*K:(i+2)*K], inversion_comp['U'][i].T)
                        #up = -np.dot( U, test_inv[ (i+1)*K:(i+2)*K, (i+1)*K:(i+2)*K])
                        
                        #test_inv[ (i+0)*K:(i+1)*K, (i+0)*K:(i+1)*K] = la.lu_solve((LU, piv),  np.eye(K) ) - np.dot( U, up.T )
                        #test_inv[ (i+0)*K:(i+1)*K,  (i+1)*K:(i+2)*K] = up
                        #test_inv[ (i+1)*K:(i+2)*K,  (i+0)*K:(i+1)*K] = up.T
                        if rhs_diag_sub:
                            solution[(i+1)*K:(i+2)*K, : ] += np.dot(up.T, btd_inference.transpose_submatrices(rhs_Ck(i),rhs_block_num,rhs_block_width))
                            solution[(i+0)*K:(i+1)*K, : ] = la.lu_solve((LU, piv),  rhs_Dk(i) ) - np.dot( np.dot( up, U.T ), rhs_Dk(i) ) + np.dot(up, rhs_Ck(i))
                        else:
                            solution[(i+0)*K:(i+1)*K, : ] = la.lu_solve((LU, piv),  rhs_Dk(i) ) - np.dot( np.dot( up, U.T ), rhs_Dk(i) )
                            
                        prev_Uup = np.dot(U, up.T)
                
                # solution with front multiplier
                if use_front_multiplier and (i != N-1): # compute front multiplier solution for the solution from previous iteration
                    front_mult_solution[(i+1)*front_multiplier.shape[0]:(i+2)*front_multiplier.shape[0],:] = np.dot(front_multiplier, solution[(i+1)*K:(i+2)*K, : ] )
                
                # solution trace                    
                if comp_solution_trace and (i != N-1): # compute trace for the solution from previous iteration
                    tr_tmp = solution[(i+1)*K:(i+2)*K, : ].reshape(K,rhs_block_num,K).swapaxes(1,0)  
                    solution_trace[i+1,:] = tr_tmp.trace(axis1=1,axis2=2)
        
        # compute lacking trace computation from the last step
        if comp_solution_trace:
            tr_tmp = solution[(0)*K:(1)*K, : ].reshape(K,rhs_block_num,K).swapaxes(1,0)  
            solution_trace[0,:] = tr_tmp.trace(axis1=1,axis2=2)
            
        # compute lacking front multiplier solution computation from the last step
        if use_front_multiplier: # compute front multiplier solution for the solution from previous iteration
            front_mult_solution[(0)*front_multiplier.shape[0]:(1)*front_multiplier.shape[0],:] = np.dot(front_multiplier, solution[(0)*K:(1)*K, : ] )
        
        # logdet, solution, trace, inversion_comp
        #import pdb; pdb.set_trace()
        
        if comp_matrix_determinant:
            r1 = logdetA.sum() if comp_matrix_determinant else None
        else:
            r1= None
        
        if comp_solution:
            if use_front_multiplier:
                r2 = front_mult_solution
            else:
                r2 = solution
        else:
            r2 = None
        
        r3 = solution_trace.sum(axis=0) if comp_solution_trace else None
        
        return r1,r2,r3,inversion_comp
    
    @staticmethod
    def transpose_submatrices(M, submatrix_number, S, row=True):
        """
        if row== True, in a matrix of shape (K, K*submatrix_number) (which is stack of submatrix_number matrics)
        transpose every submatrix. 
        
        If row = False, is a matrix shape (K*submatrix_number, K) transpose every submatrix.
        
        Input:
        ------------------
        
        M: array
            Matrix
        
        submatrix_number: int
            Number of submatrices of width S in M.
            
        S: int
            Block width in the stacked submatrices.
        
        row: bool
            If true it is assumed that submatrices are stacked row-wise.
            If false it is assumed that they are stacked columnwise.
        """
        
        #import pdb; pdb.set_trace()
        
        if submatrix_number > 1:
            R = M.copy()
            
            if row == True: # matrices stacked row wise ( the shape is (K, S*submatrix_number) )
                K = M.shape[0] # block size
                R = R.reshape(K,submatrix_number,S).transpose( (2,1,0 ) ).reshape( (S,submatrix_number*K) )
            else: # matrices stacked row wise ( the shape is (S*submatrix_number, K) )
                K = M.shape[1] # block size
                R = np.rollaxis(R.reshape((submatrix_number,S,K)), 2,1 ).reshape((submatrix_number*K,S))
        elif submatrix_number == 1:
            R = M.copy().T
        else:
            raise ValueError("transpose_submatrices: Wrong submatrix number") 
            
        return R
        
    @staticmethod 
    def btd_times_vector(matr_D, vect, matr_low_D=None, grad_params_no=None, Kv=None):
        """
        Function computes the product of SYMMETRIC btd matrix and a vector.
        BTD matrix is passed as matr_D and matr_low_D       
        
        Actually symmetricity only wrt block lower diagonals. The diagonal blocks
        may not be symmetric. If there are no matr_low_D, the matrix may be non-symmetric         
        
        Input:
        --------------------
        
        matr_D: matrix[N*Kv, K*grad_params_no]
            Block diagonal
            
        matr_low_D: matrix[(N-1)*Kv, K*grad_params_no]
            Lower diagonal            
            
        vect: matrix[N*K,grad_params_no]
        
        grad_params_no: int
            Number of derivative blocks.
        
        Kv: int
            Vertical block size. It might be different from block size K when
            there are no matr_low_D.
        """
        #import pdb; pdb.set_trace()
        
        if grad_params_no is None:
            grad_params_no = 1

        K =  int(matr_D.shape[1] / grad_params_no) # the width of the matrix block
        if Kv is None:
            Kv = K # the vertical block size. 
        else:
            if Kv != K:
                assert matr_low_D is None, "Vertical block size different from the regular block size onlu when low diagonal is None."
                
        N = int( matr_D.shape[0]/Kv)        
            
        vect = vect.reshape((N,K)).repeat(Kv,axis=0)
        
        res = np.empty( (N*Kv, grad_params_no) )        
        
        for i in range(0,grad_params_no):
            
            res[:,i] = np.sum( matr_D[:, i*K:(i+1)*K]*vect , axis=1) # diagonal part
            
            if matr_low_D is not None:
                res[K:,i] += np.sum(matr_low_D[:, i*K:(i+1)*K]*vect[0:-K, :], axis=1)
                res[0:-K,i] += np.sum( btd_inference.transpose_submatrices(matr_low_D[:, i*K:(i+1)*K], N-1, K, row=False)*vect[K:,:], axis=1 )
                
        return res
        
    @staticmethod
    #@profile
    def test_build_matrices(X, Y, F, L, Qc, P_inf, H, p_largest_cond_num, p_regularization_type=2, 
                       compute_derivatives=False, dP_inf=None, dF=None, dQc=None):    
        """
        Test the build_matrices function. This function does the same coputations as build_matrices
        but in a sparse matrix form which is much more transparent.
        Result should coincide.         
        """
        #import pdb; pdb.set_trace() 
        block_size = F.shape[0]
        x_points_num = X.shape[0]
        grad_params_no = dF.shape[2]
        if not isinstance(p_largest_cond_num, float):
            raise ValueError('sparse_inference.sparse_inverse_cov: p_Inv_jitter is not float!')
            
        dt = np.diff(X,axis=0)
        if np.any(dt < 1e-3):
            raise ValueError("btd_inference.build_matrices: small dt explore!")
            
        Ai_size = x_points_num * block_size
        Ait = np.zeros( (Ai_size, Ai_size))
        Qi = np.zeros( (Ai_size, Ai_size) )
        
        Ait_derivatives = np.zeros( (grad_params_no, Ai_size, Ai_size) )
        Qi_derivatives = np.zeros( (grad_params_no, Ai_size, Ai_size) )
                #(self, F,L,Qc,dt,compute_derivatives=False, grad_params_no=None, P_inf=None, dP_inf=None, dF = None, dQc=None)
        AQcomp = btd_inference.AQcompute_batch_Python(F,L,Qc,dt, compute_derivatives, grad_params_no, P_inf, dP_inf, dF, dQc)
        
        b_ones = np.eye(block_size)
        
        # Fill matrices ->
        #import pdb; pdb.set_trace()
        P_inf = 0.5*(P_inf + P_inf.T)
                
        (U,S,Vh) = la.svd(P_inf, compute_uv=True,)
        P_inf_inv, new_S,_ = btd_inference.psd_matrix_inverse(0, P_inf, U=None,S=None,p_largest_cond_num=p_largest_cond_num, regularization_type=p_regularization_type)
        # Different inverse computation >-
        Ait[0:block_size,0:block_size] = b_ones
        Qi[0:block_size,0:block_size] = P_inf_inv
        
        for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
            Qi_derivatives[dd,0:block_size,0:block_size] = -np.dot(P_inf_inv, np.dot(dP_inf[:,:,dd], P_inf_inv))
            
        #import pdb; pdb.set_trace() 
        for k in range(0,x_points_num-1):
            Ak = AQcomp.Ak(k,None,None)
            Qk = AQcomp.Qk(k)
            Qk = 0.5*(Qk + Qk.T) # symmetrize because if Qk is not full rank it becomes not symmetric due to numerical issues
            #import pdb; pdb.set_trace()
            Qk_inv, new_S = AQcomp.Q_inverse(k, p_largest_cond_num, p_regularization_type) # in AQcomp numbering starts from 0, consequence of Python indexing.
            if np.any((np.abs(Qk_inv - Qk_inv.T)) > 0):
                raise ValueError('sparse_inverse_cov: Qk_inv is not symmetric!')
            
            row_ind_start = (k+1)*block_size
            row_ind_end = row_ind_start + block_size
            col_ind_start = k*block_size
            col_ind_end = col_ind_start + block_size
            
            Ait[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = -Ak.T
            Ait[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = b_ones        
            Qi[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = Qk_inv
            
            dAk = AQcomp.dAk(k)
            dQk = AQcomp.dQk(k)
            
            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
                dAk_p = dAk[:,:,dd]
                dQk_p = dQk[:,:,dd]
                dQk_p = 0.5*(dQk_p + dQk_p.T)
     
                Ait_derivatives[dd, col_ind_start:col_ind_end, row_ind_start:row_ind_end] = -dAk_p.T
                Qi_derivatives[dd, row_ind_start:row_ind_end, row_ind_start:row_ind_end] = -np.dot( Qk_inv, np.dot( dQk_p, Qk_inv) )
        # Fill matrices <-
        
        #import pdb; pdb.set_trace() 
        # Perform matrix multiplication ->        
        Ki = np.dot( Ait, np.dot(Qi, Ait.T) )
        dKi = np.empty( ((grad_params_no, Ai_size, Ai_size) ) )
        for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
            tmp0 = np.dot( Ait_derivatives[dd,:,:], np.dot(Qi, Ait.T ) )
            dKi[dd,:,:] = tmp0 + tmp0.T + np.dot( Ait, np.dot(Qi_derivatives[dd,:,:] ,Ait.T ) )
        # Perform matrix multiplication <-
                
        # Read off the main block-diagonal and sub-diagonal ->
        Ki_diag = np.empty( (x_points_num*block_size, block_size) )
        Ki_low_diag = np.empty( (block_size*(x_points_num-1), block_size) )
        _, Ki_logdet = np.linalg.slogdet(Ki)
        
        d_Ki_diag = np.empty( (block_size*x_points_num, block_size*grad_params_no) )
        d_Ki_low_diag = np.empty( (block_size*(x_points_num-1), block_size*grad_params_no) )
        d_Ki_logdet = np.empty((grad_params_no,) )
        
        #import pdb; pdb.set_trace()
        Ki_diag[0:block_size] = Ki[0:block_size, 0:block_size]
        for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
            d_Ki_diag[0:block_size,(dd+0)*block_size:(dd+1)*block_size] = dKi[dd,0:block_size,0:block_size]  
            d_Ki_logdet[dd] = np.trace( np.linalg.lstsq( Ki, dKi[dd,:,:])[0] )
            
        for k in range(0,x_points_num-1):
            Ki_diag[(k+1)*block_size:(k+2)*block_size] = Ki[(k+1)*block_size:(k+2)*block_size, (k+1)*block_size:(k+2)*block_size]
            Ki_low_diag[(k+0)*block_size:(k+1)*block_size] = Ki[(k+1)*block_size:(k+2)*block_size, (k+0)*block_size:(k+1)*block_size]
            
            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
                d_Ki_diag[(k+1)*block_size:(k+2)*block_size,(dd+0)*block_size:(dd+1)*block_size] = \
                    dKi[dd, (k+1)*block_size:(k+2)*block_size, (k+1)*block_size:(k+2)*block_size]  
                d_Ki_low_diag[(k+0)*block_size:(k+1)*block_size, (dd+0)*block_size:(dd+1)*block_size] = \
                    dKi[dd, (k+1)*block_size:(k+2)*block_size, (k+0)*block_size:(k+1)*block_size]
                
        # Read off the main block-diagonal and sub-diagonal <-
        
        return Ki_diag, Ki_low_diag, Ki_logdet, d_Ki_diag, d_Ki_low_diag, d_Ki_logdet, Ki, dKi
                
    @staticmethod
    #@profile
    def build_matrices(X, Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, p_regularization_type=2, 
                       compute_derivatives=False, dP_inf=None, dP0 = None, dF=None, dQc=None,
                       also_sparse=False):
        """
        TODO: introduce P0 for covariance and P_inf for noise calculation.
        K - block size
        
        Input:
        ----------------------------------
        
        dP_inf: matrix[K,K,  grad_params_no ]
            Derivatives of P_inf. Derivatives along the 3-rd dimension.            
        
        
        
        also_sparse: bool
            Return also sparse BTD lil matrix. Later derivatives might be added.
            
        Output:
        ----------------------------------
            Ki_D: matrix(N*K, K)
                Diagonal of the Ki matrix, diagonal blocks are in block-column
            Ki_C: matrix((N-1)*K, K)
                Subdiagonal of he Ki matrix, subdiagonal blocks are in block-column
            Ki_DN: matrix(N*K, K)
                The same as Ki_D plus HtH to each block
            d_Ki_D: matrix(N*K, K*deriv_num)
                Derivatives of the matrix Ki_D, derivatives are row-wise
            d_Ki_C: matrix((N-1)*K, K*deriv_num)
                Derivatives of the matrix Ki_C, derivatives are row-wise
            
        """
        #import pdb; pdb.set_trace()
        K = F.shape[0] # block_size
        N = X.shape[0]
        
        # Allocating space ->
        Ki_diag = np.empty( (K*N, K) )
        Ki_low_diag = np.empty( (K*(N-1), K))  
        Ki_logdet = np.empty((N,))
        
        if compute_derivatives:
            grad_params_no = dF.shape[2]
            
            d_Ki_diag = np.empty( (K*N, K*grad_params_no) )
            d_Ki_low_diag = np.empty( (K*(N-1), K*grad_params_no) )
            d_Ki_logdet = np.empty((N,grad_params_no))
        else:
            grad_params_no = 1
            d_Ki_diag = None
            d_Ki_low_diag = None
            d_Ki_logdet = None
        # Allocating space <-
        
        # dt handling ->
        dt = np.diff(X,axis=0)
        if np.any(dt < 1e-3):
            raise ValueError("btd_inference.build_matrices: small dt explore!")
        
        #import pdb; pdb.set_trace()
        P_inf = 0.5*(P_inf + P_inf.T)
        _, _,P_inv_reg = btd_inference.psd_matrix_inverse(0, P_inf, U=None,S=None, p_largest_cond_num=p_largest_cond_num,  regularization_type =p_regularization_type) #=p_largest_cond_num, regularization_type=p_regularization_type)
        
        AQcomp = btd_inference.AQcompute_batch_Python(F,L,Qc,dt, compute_derivatives, grad_params_no, P_inv_reg, dP_inf, dF, dQc)
        # dt handling <-
        
        #import pdb; pdb.set_trace()
        # First diagonal block computation ->
        P0 = 0.5*(P0 + P0.T)
        #P0_inv, new_S,_ = btd_inference.psd_matrix_inverse(0, P0, U=None,S=None,p_largest_cond_num=p_largest_cond_num, regularization_type=p_regularization_type)
        P0_inv, new_S,_ = btd_inference.psd_matrix_inverse(0, P0, U=None,S=None,p_largest_cond_num=p_largest_cond_num, regularization_type = p_regularization_type ) #=p_largest_cond_num, regularization_type=p_regularization_type)
        
        Ki_diag[0:K,:] = P0_inv
        Ki_logdet[0] = np.sum( np.log( new_S[ new_S > 0.0] ) )
        
        # Filling sparse matrix ->
        if also_sparse:
            Ki_sparse = sparse.lil_matrix( (N*K, N*K) )
            Ki_sparse[0:K,0:K] = P0_inv
        # Filling sparse matrix <-
            
        #import pdb; pdb.set_trace()
        if compute_derivatives:
            # Derivative are of shape Dir[K,K, grad_params_no] - along the third dimension.            
            # 1) Multiply each derivative by matrix A in front: np.dot(A, np.rollaxis(Dir,1))
            # the result is of the same shape [K,K, grad_params_no]
            
            # 2) Multiply each derivative by matrix A from back. Deriv. matrices as before of
            # Dir[K,K, grad_params_no], operation: np.dot(np.rollaxis(Dir,2), A)
            # the result is of the shape [grad_params_no,K,K]
            
            # 3) Shape Dir[grad_params_no,K,K] multiply from the BACK by matrix A[K,K]:
            #    np.dot(Dir,A). Resulting shape is the same: [grad_params_no,K,K]
            
            # 4) Shape Dir[grad_params_no,K,K] multiply from the FRONT by matrix A[K,K]:
            #    np.dot( A, np.transpose(Dir,(2,1,0)) ). New shape is [K,K, grad_params_no].
            
            # 5) From the Dir[grad_params_no,K,K] shape of derivative matrix make
            # [K,k*grad_params_no]: np.rollaxis(Dir,1).reshape(K,K*grad_params_no)
            
            # 6) From the Dir[K,K,grad_params_no] shape of derivative matrix make
            # [K,k*grad_params_no]: Dir.transpose((0,2,1)).reshape(K,K*grad_params_no)
            
            # 7) From shape Dir[K,K,grad_params_no] to shape Dir[grad_params_no,K,K]:
            #    np.rollaxis(Dir,2)
            
            # 8) From Dir[grad_params_no,K,K] to the right shape Dir[K,K*grad_params_no]:
                # np.rollaxis(Dir,1).reshape(K,K*grad_params_no)
            
            # 9) From Dir[K,K,grad_params_no] to the right shape Dir[K,K*grad_params_no]:
            #    Dir.transpose((0,2,1)).reshape(K,K*grad_params_no)
            
            # Implement formula: d{Pi^{-1}}/dp = -Pi^{-1} d{Pi}/dp Pi^{-1}
            # 0-th diagonal block computation ->
            #tmp0 = -np.dot(P_inf_inv, np.rollaxis(dP_inf,1) ) # shapse is [K,K, grad_params_no]
            tmp0 = -np.dot( np.rollaxis(dP0,2), P0_inv ) # shape is [grad_params_no,K,K]
            
            #tmp1 = np.dot(np.rollaxis(tmp0,2), P_inf_inv) # shapse is [grad_params_no,K,K]
            tmp1 = np.dot( P0_inv, np.transpose(tmp0,(2,1,0)) ) # shape is [K,K, grad_params_no]
            
            #d_Ki_diag[0:K,:] = np.rollaxis(tmp1,1).reshape(K,K*grad_params_no)
            d_Ki_diag[0:K,:] = tmp1.transpose((0,2,1)).reshape(K,K*grad_params_no)
            # 0-th diagonal block computation <-
            
            d_Ki_logdet[0,:] = np.trace(tmp0, axis1=1, axis2=2)
        # First diagonal block computation <-
            
        #import pdb; pdb.set_trace()
        for k in range(0,N-1):
            Ak = AQcomp.Ak(k,None,None)
            Qk = AQcomp.Qk(k)
            Qk = 0.5*(Qk + Qk.T) # symmetrize because if Qk is not full rank it becomes not symmetric due to numerical problems
            #import pdb; pdb.set_trace()
            #Qk_inv, new_S = AQcomp.Q_inverse(k, p_largest_cond_num, p_regularization_type) # in AQcomp numbering starts from 0, consequence of Python indexing.
            Qk_inv, new_S = AQcomp.Q_inverse(k, p_largest_cond_num, p_regularization_type) # in AQcomp numbering starts from 0, consequence of Python indexing.
            
            Ki_low_diag[(k)*K:(k+1)*K, :] = - np.dot( Qk_inv, Ak )
            
            Ki_diag[ (k)*K:(k+1)*K, :] += np.dot( Ak.T, np.dot( Qk_inv, Ak ) ) # update previous diagonal element
            Ki_diag[ (k+1)*K:(k+2)*K, :] = Qk_inv
            
            # Filling sparse matrix ->
            if also_sparse:
                Ki_sparse[ (k+1)*K:(k+2)*K, (k)*K:(k+1)*K ] = Ki_low_diag[(k)*K:(k+1)*K, :] # Ki_low_diag
                Ki_sparse[ (k)*K:(k+1)*K, (k+1)*K:(k+2)*K ] = Ki_low_diag[(k)*K:(k+1)*K, :].T # Ki_high_diag 
                
                Ki_sparse[(k)*K:(k+1)*K,(k)*K:(k+1)*K] += Ki_low_diag[(k)*K:(k+1)*K, :] # this diag
                Ki_sparse[(k+1)*K:(k+2)*K,(k+1)*K:(k+2)*K] = Ki_diag[ (k+1)*K:(k+2)*K, :]
            # Filling sparse matrix <-
            
            # Log determinant computation ->
            if new_S is None:
                Ki_logdet[k+1] = Ki_logdet[k]
            else:
                Ki_logdet[k+1] = np.sum( np.log( new_S[ new_S > 0.0] ) )            
            # Log determinant computation <-
        
            if compute_derivatives:
                dAk = AQcomp.dAk(k) # Direvative are along the 3-d dimension e.g Ak[K,K,grad_params_no]
                dQk = AQcomp.dQk(k) # Direvative are along the 3-d dimension Qk[K,K,grad_params_no]
                
                dQk = 0.5*( dQk + np.rollaxis(dQk, 1 ) ) # Symmetrize dQk, checked
                # Current iteration diagonal ->
                #tmp0 = -np.dot(Qk_inv, np.rollaxis(dQk,1) ) # shape [K,K, grad_params_no]
                tmp0 = -np.dot( np.rollaxis(dQk,2), Qk_inv ) # shape is [grad_params_no,K,K] (Rule 6)
                #tmp1 = np.dot(np.rollaxis(tmp0,2), Qk_inv) # shape [grad_params_no,K,K], contains 
                tmp1 = np.dot( Qk_inv, np.transpose(tmp0,(2,1,0)) ) # shape is [K,K, grad_params_no] (Rule 4)
                    # dQ^{-1} / dp
                #d_Ki_diag[ (k+1)*K:(k+2)*K, :] = np.rollaxis(tmp1,1).reshape(K,K*grad_params_no)
                d_Ki_diag[ (k+1)*K:(k+2)*K, :] = tmp1.transpose((0,2,1)).reshape(K,K*grad_params_no) # shape is 
                # Current iteration diagonal <-
                
                # Derivatives of Log determinant computation ->
                d_Ki_logdet[k+1,:] = np.trace(tmp0, axis1=1, axis2=2)
                # Derivatives of Log determinant computation <-
                
                # This iteration off-diagonal ->
                # tmp1 contains d{Qi^{-1}}/dp = -Qi^{-1} d{Qi}/dp Qi^{-1} of shape [grad_params_no,K,K]
                #tmp2 = np.dot(tmp1, Ak) # shape of result is [grad_params_no,K,K]
                tmp2 = np.dot(np.rollaxis(tmp1,2), Ak) # shape of result is [grad_params_no,K,K]
                tmp3 = np.dot(Qk_inv, np.rollaxis(dAk,1)) # shape of result is [K,K,grad_params_no] (Rule 5)
                
                tmp2 = np.rollaxis(tmp2,1).reshape(K,K*grad_params_no)
                tmp3 = tmp3.transpose((0,2,1)).reshape(K,K*grad_params_no)
                
                d_Ki_low_diag[(k)*K:(k+1)*K, :] = -1*(tmp2+tmp3)
                # This iteration off-diagonal <-
                
                # Update previous iteration diagonal ->
                tmp4 = np.dot(Ak.T,tmp3)
                d_Ki_diag[ (k)*K:(k+1)*K, :] += np.dot(Ak.T,tmp2) + tmp4 + \
                   btd_inference.transpose_submatrices(tmp4, grad_params_no, K, row=True)
                # Update previous iteration diagonal <-
            else:
                d_Ki_diag = None
                d_Ki_low_diag = None
                d_Ki_logdet = None
        if not also_sparse:
            return Ki_diag, Ki_low_diag, Ki_logdet.sum(), d_Ki_diag if d_Ki_diag is not None else None, \
                d_Ki_low_diag, d_Ki_logdet.sum(axis=0) if d_Ki_logdet is not None else None
        else:
            return Ki_diag, Ki_low_diag, Ki_logdet.sum(), d_Ki_diag if d_Ki_diag is not None else None, \
                d_Ki_low_diag, d_Ki_logdet.sum(axis=0) if d_Ki_logdet is not None else None, Ki_sparse
        
    def test_marginal_ll(Y, Ki, Ki_logdet, H, g_noise_var, 
                    compute_derivatives=False, d_Ki=None, d_Ki_logdet = None):
        """
        Input:
        ------------------
        """
        #import pdb; pdb.set_trace()
        N = Y.shape[0] # number of data points
        
        g_noise_var = float(g_noise_var)
        
        HtH = np.dot(H.T, H)
        Gt = np.kron( np.eye(N), H.T )
        GtY = np.dot( Gt, Y)        
        
        M = Ki + np.kron( np.eye(N), HtH) / g_noise_var # G^{T} \Sigma^{-1} G
        _, mll_log_det = np.linalg.slogdet(M)
        mll_log_det += (Y.size)*np.log(g_noise_var) - Ki_logdet
        
        mll_data_fit_term = np.dot(Y.T, Y)/g_noise_var - np.dot(GtY.T, np.linalg.solve( M , GtY ) )/g_noise_var**2
        marginal_ll = -0.5*(mll_data_fit_term + mll_log_det + (Y.size)*log_2_pi)
        
        #import pdb; pdb.set_trace()
        if compute_derivatives:
            grad_params_no = d_Ki.shape[0]                
            
            mll_data_fit_deriv = np.empty( (grad_params_no+1,1) ) # including noise
            mll_determ_deriv = np.empty( (grad_params_no+1,1) ) # including noise
            
            tmp0 = np.linalg.solve(M , GtY)
            for k in range(0, grad_params_no):
                
                mll_data_fit_deriv[k,0] = np.dot( tmp0.T, np.dot( d_Ki[k,:,:], tmp0 ) )/ g_noise_var**2
                
                mll_determ_deriv[k,0] = np.trace( np.linalg.solve( M, d_Ki[k,:,:] ) ) - d_Ki_logdet[k] + 0
            
            tmp1 = np.dot(Gt.T, tmp0)
            
            mll_data_fit_deriv[-1,0] = -np.dot(Y.T,Y)/ g_noise_var**2 + 2.0*np.dot( GtY.T, tmp0)/g_noise_var**3 - \
                                        np.dot(tmp1.T, tmp1)/g_noise_var**4
                                     
                    # derivative wrt noise
            mll_determ_deriv[-1,0] = -np.trace( np.linalg.solve( M, np.dot(Gt,Gt.T) ) )/g_noise_var**2 + N/g_noise_var  # derivative wrt noise
            
            d_marginal_ll = -0.5*( mll_data_fit_deriv + mll_determ_deriv)
        
        #import pdb; pdb.set_trace()
        return marginal_ll, d_marginal_ll, mll_data_fit_term, mll_log_det, mll_data_fit_deriv, mll_determ_deriv
        
    @staticmethod
    def marginal_ll(K, Y, Ki_diag, Ki_low_diag, Ki_logdet, H, g_noise_var, 
                    compute_derivatives=False, d_Ki_diag=None, d_Ki_low_diag=None,
                    d_Ki_logdet = None):
                
        """
        Function computes marginal likelihood and its derivatives.
        
        Input:
        --------------------
        
        K: int
            Block size
        Y matrix[N,1]
            Measurements as a [N,1] column
        """
        #import pdb; pdb.set_trace()
        N = Y.shape[0] # number of data points
        
        g_noise_var = float(g_noise_var)
        
        HtH = np.dot(H.T, H)        
        KiN_diag = Ki_diag +  np.tile( HtH/g_noise_var, (N,1) )       
        
        GtY =  np.tile(H.T, (N,1)) * Y.repeat(K, axis=0)       
        
        KiN_logdet, KiNiGtY, _, inc_com = \
            btd_inference.block_tridiag_solver(N,K, KiN_diag, Ki_low_diag, GtY, rhs_diag=False, inversion_comp=None, rhs_C_matr=None,
                             comp_matrix_determinant=True, comp_solution=True, comp_solution_trace=False)
        
        mll_log_det = -Ki_logdet# ignoring 0.5 and - sign.
        mll_log_det += KiN_logdet
        mll_log_det += (Y.size)*np.log(g_noise_var)
        

        tmp1 = np.dot(GtY.T, KiNiGtY)
        mll_data_fit_term = ( np.dot(Y.T,Y) /g_noise_var - tmp1/g_noise_var**2 ) # ignoring 0.5 and - sign.        
        marginal_ll = -0.5 * ( mll_log_det + mll_data_fit_term + (Y.size)*log_2_pi)
        
        #import pdb; pdb.set_trace()
        d_marginal_ll = None
        if compute_derivatives:
            grad_params_no = int( d_Ki_diag.shape[1]/K )
            
            mll_data_fit_deriv = np.empty( (grad_params_no+1,1) ) # including noise
            mll_determ_deriv = np.empty( (grad_params_no+1,1) ) # including noise
            
            noise_square = g_noise_var**2            
            
            tmp3 = btd_inference.btd_times_vector( d_Ki_diag,  KiNiGtY, d_Ki_low_diag, grad_params_no )
            
            mll_data_fit_deriv[0:-1,0] = np.dot( KiNiGtY.T, tmp3 ) / noise_square # without noise derivative.
            
            tmp4 = btd_inference.btd_times_vector( np.tile(H, (N,1)),  KiNiGtY, None, 1, 1 )
            mll_data_fit_deriv[-1,0] = 1.0/noise_square* ( -np.dot( Y.T, Y) + 2.0* tmp1/g_noise_var -
                np.dot(tmp4.T, tmp4)/noise_square)
            
            d_Ki_diag = np.hstack( (d_Ki_diag, np.tile( HtH, (N,1))) )# add noise related part to calculate noise drivative as well
                # ignore 1/g_noise_var for now
            d_Ki_low_diag = np.hstack( (d_Ki_low_diag, np.zeros( ((N-1)*K, K)) ) )
            
            # def block_tridiag_solver(N,K, A_matr, C_matr, rhs_D_matr, rhs_diag=False, inversion_comp=None, rhs_C_matr=None,
            #                 comp_matrix_determinant=False, comp_solution=False, comp_solution_trace=False,
            #                 rhs_block_num=None, rhs_block_width = None, front_multiplier=None)
                             
            (_,_,tmp5,_) = btd_inference.block_tridiag_solver(N,K, KiN_diag, Ki_low_diag, d_Ki_diag, rhs_diag=True, 
                                                      inversion_comp=inc_com,  rhs_C_matr=d_Ki_low_diag,
                             comp_matrix_determinant=False, comp_solution=False, comp_solution_trace=True, rhs_block_num=grad_params_no+1, rhs_block_width = K)
            
            # The first determinant derivative: logdet( K1^{-1} + Gt \Sigma G )
            mll_determ_deriv[:,0] = tmp5;  mll_determ_deriv[-1,0] = -mll_determ_deriv[-1,0]/g_noise_var**2 # take into account earlier ignored division
            mll_determ_deriv[:-1,0] -= d_Ki_logdet
            mll_determ_deriv[-1,0] += N/g_noise_var 
            
            d_marginal_ll = -0.5*( mll_determ_deriv + mll_data_fit_deriv)
        else:
            mll_data_fit_deriv = None
            mll_determ_deriv = None
            
        return marginal_ll, d_marginal_ll, mll_data_fit_term, mll_log_det, mll_data_fit_deriv, mll_determ_deriv
    
    
    @staticmethod
    def mean_var_calc_prepare_matrices(K, X_train, X_test, Y_train, var_or_likelihood, F, L, Qc, P_inf, P0, H,
                                       p_largest_cond_num=1e+13, p_regularization_type=2, diff_x_crit=None):
        
        """
        
        Input:
        ---------------------
        
        p_regularization_type: 1 or 2
        
        
        p_largest_cond_num: float
            Largest condition number of the Qk matices. See function 
            "sparse_inverse_cov".
        
        
        
        diff_x_crit: float (not currently used)   
            If new X_test points are added, this tells when to consider 
            new points to be distinct. If it is None then the same variable
            is taken from the class.
            
        """
        
        if diff_x_crit is not None: # currently not implememted
            pass
            
        #import pdb; pdb.set_trace()  
        if (X_test is None) or (X_test is X_train) or ( (X_test.shape == X_train.shape) and np.all(X_test == X_train) ):
            # Consider X_test is equal to X_train
            X_test = None            
            test_points_num = None
            
        if X_test is not None:
            test_points_num = X_test.shape[0]
            
            X = np.vstack((X_train, X_test))
            Y = np.vstack((Y_train, np.zeros(X_test.shape)) )
            
            which_train = np.vstack( ( np.ones( X_train.shape), np.zeros( X_test.shape)) )
            
            _, return_index, inverse_index = np.unique(X,True,True)
             
            X = X[return_index]
            Y = Y[return_index]
            which_train = which_train[return_index]
            
        else:
            X = X_train
            Y = Y_train
            which_train = np.ones(X_train.shape)
        
            inverse_index = None
            return_index = None
                       
        Ki_diag, Ki_low_diag,_ ,_ ,_ ,_ = btd_inference.build_matrices(X, Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, p_regularization_type=2, 
                       compute_derivatives=False, dP_inf=None, dP0= None, dF=None, dQc=None)
        
        return Ki_diag, Ki_low_diag, test_points_num, return_index, inverse_index
        
    @staticmethod
    def mean_var_calc(K, Y, Ki_diag, Ki_low_diag, H, g_noise_var, test_points_num, forward_index, inverse_index):
        """
        Compute mean and variance of gaussian process.
        
        Input:
        ------------------
        
        K: int
            Block Size
            
        Y: vector(N,1)
             Original training points
        
        which_observed: vector(N,)
            Vector of the same size as Y where 1 indicates that this is a trainig point
            while 0 indicates that this is a test point.             
             
        Ki_diag: matrix( N*K, K)
            BTD diagonal
            
        Ki_low_diag: matrix
        
        H:
        
        g_noise_var:
    
        """
        # !!! TODO: P0 and P_inf handle
        if forward_index is not None:
            which_train = np.vstack( ( np.ones(Y.shape), np.zeros( (test_points_num,1) )) )
            which_train = which_train[forward_index]
            
            Y = np.vstack(  ( Y, np.zeros((test_points_num,1)) )  )
            Y = Y[forward_index]
        else:
            which_train = np.ones(Y.shape)
            test_points_num = Y.shape[0] # output all points
             
        which_train_repeated = np.repeat(which_train, K, axis=0)
        Y_repeated = np.repeat(Y, K, axis=0)
        
        N = Y.shape[0]
        HtH = np.dot(H.T, H)        
        
        Ht_repeted = np.tile( H.T, (N,1) )        
        
        KiN_diag = Ki_diag + np.tile(HtH, (N,1)) * np.tile( which_train_repeated, (1,K) ) / g_noise_var
        rhs = Ht_repeted * Y_repeated / g_noise_var
        
        _,means,_,inversion_comp = btd_inference.block_tridiag_solver(N,K, KiN_diag, Ki_low_diag, rhs, rhs_diag=False, inversion_comp=None, rhs_C_matr=None,
                             comp_matrix_determinant=False, comp_solution=True, comp_solution_trace=False,
                             rhs_block_num=None, rhs_block_width = None, front_multiplier= None )
                             
        means = np.sum( np.tile(H, (N,1) ) * means.reshape((N,K)), axis=1)
        if inverse_index is not None:
            means = means[inverse_index]; 
        means.shape = (means.shape[0],1)
        
        if test_points_num is not None:
            means = means[-test_points_num:]        
        
        _,variances,_,_ = btd_inference.block_tridiag_solver(N, K, KiN_diag, Ki_low_diag, Ht_repeted, rhs_diag=True, inversion_comp=inversion_comp, rhs_C_matr=None,
                             comp_matrix_determinant=False, comp_solution=True, comp_solution_trace=False,
                             rhs_block_num=None, rhs_block_width = None, front_multiplier= H )
                             
        if inverse_index is not None:                     
            variances = variances[inverse_index]
        
        if test_points_num is not None:
            variances = variances[-test_points_num:] 
        
        return means, variances