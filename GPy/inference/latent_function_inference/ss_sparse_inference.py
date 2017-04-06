# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import PosteriorExact as Posterior
from ...models import state_space_main as ssm
from ... import likelihoods

from scipy import sparse
import scipy.linalg as la
import sksparse.cholmod as cholmod

import time
import numpy as np
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
        
class sparse_inference(object):
    
    @staticmethod
    #@profile
    def sparse_inverse_cov(X, Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, compute_derivatives=False,
                           grad_calc_params=None, p_regularization_type=2):
        """
        Function returns all the necessary matrices for the SpInGP inference.        
        
         Notation for matrix is: K = A0, B1 0       
                                    C1, A1, B2
                                    0 , C2, A2s
                                    
        Input:
        ------------------------------
            Exactly from furnction kern.sde()
            
            H - row vector
            
            p_largest_cond_num: float
            Largest condition number of the Qk and P_inf matices. This is needed because
            for some models the these matrices are while inverse is required.
                
        Output:
        -------------------------------
        
        Ait, 
        Qi, 
        GtY, 
        G, 
        GtG, 
        HtH, 
        Ki_derivatives,
        
        Kip: sparse(Ai_size, Ai_size)
            Block diagonal matrix. The diagonal blocks are the same as in K.
            Required for ll dorivarive computation.
            
        matrix_blocks, 
        matrix_blocks_derivatives: dictionary
            Dictionary contains 2 keys: 'dAkd' and 'dCkd'. Each contain corresponding
                derivatives of matrices Ak and Ck. Shapes dAkd[0:(N-1)] [0:(deriv_num-1),0:(block_size-1), 0:(block_size-1)]
                    dCkd[0:(N-2)][0:(deriv_num-1), 0:(block_size-1), 0:(block_size-1)]
        
        """
        block_size = F.shape[0]
        x_points_num = X.shape[0]
        
        Ai_size = x_points_num * block_size
        Ait = sparse.lil_matrix( (Ai_size, Ai_size))
        Qi = sparse.lil_matrix( (Ai_size, Ai_size) )
        
        if not isinstance(p_largest_cond_num, float):
            raise ValueError('sparse_inference.sparse_inverse_cov: p_Inv_jitter is not float!')
        
        Akd = {}
        Bkd = {}
        if compute_derivatives:
            # In this case compute 2 extra matrices dA/d(Theta) and dQ^{-1}/d(Theta)
        
            dP_inf = grad_calc_params['dP_inf']
            dF = grad_calc_params['dF']
            dQc = grad_calc_params['dQc']
            
            # The last parameter is the noice variance
            grad_params_no = dF.shape[2]
            
            
            #dP0 = grad_calc_params['dP_init']
            
            AQcomp = ssm.ContDescrStateSpace._cont_to_discrete_object(X, F, L, Qc, compute_derivatives=True,
                                     grad_params_no=grad_params_no,
                                     P_inf=P_inf, dP_inf=dP_inf, dF = dF, dQc=dQc,
                                     dt0='no dt0')
            
            Kip = sparse.lil_matrix( (Ai_size, Ai_size) )                         
            # Derivative matrices                                     
            Ait_derivatives = []
            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
                Ait_derivatives.append( sparse.lil_matrix( (Ai_size, Ai_size)) )
                
            Qi_derivatives = []                      
            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
                Qi_derivatives.append( sparse.lil_matrix( (Ai_size, Ai_size)) )
            
            Ki_derivatives = [] # Derivatives of the K = At*Qi*A

            # Determinant derivatives: (somematrices required for speeding up the computation of determ derivatives)
            dAkd = {}
            dCkd = {}            
        else:
            
            AQcomp = ssm.ContDescrStateSpace._cont_to_discrete_object(X, F, L, Qc, compute_derivatives=False,
                                     grad_params_no=None,
                                     P_inf=P_inf, dP_inf=None, dF = None, dQc=None,
                                     dt0='no dt0')
            Ait_derivatives = None
            Qi_derivatives = None
            Ki_derivatives = None
            Kip = None
            
        b_ones = np.eye(block_size)
        
        #GtY = sparse.lil_matrix((Ai_size,1))
        H_nonzero_inds = np.nonzero(H)[1]
     
        G = sparse.kron( sparse.eye(x_points_num, format='csc' ), sparse.csc_matrix(H), format='csc' )
        HtH = np.dot(H.T, H)
        GtG = sparse.kron( sparse.eye(x_points_num, format='csc' ), sparse.csc_matrix( HtH ), format='csc')    
        
        Ait[0:block_size,0:block_size] = b_ones
        #import pdb; pdb.set_trace()
        P_inf = 0.5*(P_inf + P_inf.T)
                
        #p_regularization_type=1
        #p_largest_cond_num = 1e8
        (U,S,Vh) = la.svd(P_inf, compute_uv=True,)
        P_inf_inv = ssm.psd_matrix_inverse(-1, P_inf, U=U,S=S,p_largest_cond_num=p_largest_cond_num, regularization_type=p_regularization_type)
        
        # Different inverse computation >-
                    
        Qi[0:block_size,0:block_size] = P_inf_inv
        Qk_inv_prev = P_inf_inv # required for derivative of determinant computations
        
        if compute_derivatives:
            d_Qk_inv_prev = np.empty( (grad_params_no, block_size, block_size) ) # initial derivatives of dQk_inv
            
            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
                    dP_inf_p = dP_inf[:,:,dd]            
                    d_Qk_inv_prev[dd,:,:] = -np.dot(P_inf_inv, np.dot(dP_inf_p, P_inf_inv))
                    
        for k in range(0,x_points_num-1):
            Ak = AQcomp.Ak(k,None,None)
            Qk = AQcomp.Qk(k)
            Qk = 0.5*(Qk + Qk.T) # symmetrize because if Qk is not full rank it becomes not symmetric due to numerical problems
            #import pdb; pdb.set_trace()
            Qk_inv = AQcomp.Q_inverse(k, p_largest_cond_num, p_regularization_type)
            if np.any((np.abs(Qk_inv - Qk_inv.T)) > 0):
                raise ValueError('sparse_inverse_cov: Qk_inv is not symmetric!')
            
            row_ind_start = (k+1)*block_size
            row_ind_end = row_ind_start + block_size
            col_ind_start = k*block_size
            col_ind_end = col_ind_start + block_size
            
            Ait[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = -Ak.T
            Ait[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = b_ones        
            Qi[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = Qk_inv
            
            #import pdb; pdb.set_trace()
            #GtY[row_ind_start + H_nonzero_inds,0] = Y[k+1,0]* Hnn     
            Bkd[k] = - np.dot( Ak.T, Qk_inv )
            Akd[k] = np.dot( Ak.T, np.dot( Qk_inv, Ak ) ) +  Qk_inv_prev
                
            if compute_derivatives:
                if (k == 0):
                    prev_Ak = Ak # Ak from the previous step
                    prev_diag = P_inf
                    prev_off_diag = np.dot(Ak, P_inf) # previous off diagonal (lowe part)                    
                    
                    
                    Kip[col_ind_start:col_ind_end, col_ind_start:col_ind_end] = prev_diag
                    Kip[row_ind_start:row_ind_end, col_ind_start:col_ind_end] = prev_off_diag
                    Kip[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = prev_off_diag.T
                    
                    prev_diag = np.dot( Ak, np.dot(prev_diag, prev_Ak.T )) + Qk
                    Kip[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = prev_diag
                    
                else:
                    #curr_off_diag = np.dot( Ak, np.dot(prev_off_diag, prev_Ak.T) ) + np.dot(Ak, Qk)
                    curr_off_diag = np.dot( Ak, prev_diag)
                    curr_diag = np.dot( Ak, np.dot(prev_diag, Ak.T )) + Qk
                
                    Kip[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = curr_diag
                    Kip[row_ind_start:row_ind_end, col_ind_start:col_ind_end] = curr_off_diag
                    Kip[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = curr_off_diag.T
                    
                    prev_diag = curr_diag
                
                dAkd_k = np.empty( (grad_params_no, block_size, block_size) )
                dCkd_k = np.empty( (grad_params_no, block_size, block_size) )
                
                dAk = AQcomp.dAk(k)
                dQk = AQcomp.dQk(k)
                
                for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
                    dAk_p = dAk[:,:,dd]
                    dQk_p = dQk[:,:,dd]
                    dQk_p = 0.5*(dQk_p + dQk_p.T)
         
                    sparse_dAit = Ait_derivatives[dd]
                    sparse_dQi = Qi_derivatives[dd]
                    
                    tmp1 = -np.dot(Qk_inv, np.dot(dQk_p, Qk_inv))
                    sparse_dAit[col_ind_start:col_ind_end, row_ind_start:row_ind_end] = -dAk_p.T
                    sparse_dQi[row_ind_start:row_ind_end, row_ind_start:row_ind_end] = tmp1
                    
                    dAkd_k[dd,:,:] = np.dot( dAk_p.T, np.dot( Qk_inv, Ak) )
                    dAkd_k[dd,:,:] += dAkd_k[dd,:,:].T
                    
                    tmp2 =  np.dot( Qk_inv, Ak) 
                    dAkd_k[dd,:,:] += -np.dot(tmp2.T, np.dot(dQk_p, tmp2)) + d_Qk_inv_prev[dd]
                    d_Qk_inv_prev[dd,:,:] = tmp1
                    
                    dCkd_k[dd,:,:] = -np.dot( Qk_inv, dAk_p) - np.dot(tmp1,Ak )
                    
                dAkd[k] = dAkd_k
                dCkd[k] = dCkd_k
                    
            Qk_inv_prev = Qk_inv # required for derivative of determinant computations      
                      
        
        Qi = Qi.asformat('csc')
        Ait = Ait.asformat('csc')
        #Bkd[k] = - np.dot( Ak.T, Qk_inv )
        
        Akd[x_points_num-1] = Qk_inv_prev # set the last element of the matrix        
        if compute_derivatives:                 
            dAkd[x_points_num-1] = d_Qk_inv_prev # set the last element of the matrix
            
            for dd in range(0,grad_params_no): # ignore derivatives wrt noise variance
                dP_inf_p = dP_inf[:,:,dd]            
            
                sparse_dQi = Qi_derivatives[dd]
                sparse_dQi[0:block_size,0:block_size] = -np.dot(P_inf_inv, np.dot(dP_inf_p, P_inf_inv))
                
                sparse_dQi = sparse_dQi.asformat('csc')
                sparse_dAit = Ait_derivatives[dd]
                sparse_dAit = sparse_dAit.asformat('csc')                
                
                Ki_der = (sparse_dAit * Qi) * Ait.T # TODO: maybe this is block matrix
                Ki_der += Ki_der.T
                Ki_der += (Ait*sparse_dQi)*Ait.T                
                
                Ki_derivatives.append(Ki_der)
                                
         
        GtY = G.T * sparse.csc_matrix(Y)
        #GtY[H_nonzero_inds,0] = Y[0,0]* Hnn     # insert the first block
        GtY = GtY.asformat('csc')  
        #GtY = sparse.kron( Y, H.T, format='csc') # another way to compute
        
        
        matrix_blocks = {}
        matrix_blocks_derivatives = {}
        matrix_blocks['Akd'] = Akd
        matrix_blocks['Bkd'] = Bkd
        if compute_derivatives:
            matrix_blocks_derivatives['dAkd'] = dAkd
            matrix_blocks_derivatives['dCkd'] = dCkd
        
        return Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, matrix_blocks_derivatives

    @staticmethod
    def marginal_ll(block_size, Y, Ait, Qi, GtY, G, GtG, H, g_noise_var, 
                    compute_derivatives=False, dKi_vector=None, Kip=None,
                    matrix_blocks=None, matrix_blocks_derivatives=None):
        """
        Function computes  marginal likelihood and its derivatives.        
        
        Inputs are mostly the necessary matrices obtained from the function "sparse_inverse_cov".
        Input:        
        -----------------------
        compute_inv_main_diag: bool
            Whether to compute intermidiate data for inversion of sparse
            tridiagon precision. This is needed for further variance calculation.
            For marginal likelihood and its gradient it is not required.
        
        Kip: sparce
            Block diagonal matrix. The diagonal blocks are the same as in K.
            Required for ll derivarive computation.
        """
        measure_timings = True        
        
        
        if measure_timings:
            meas_times = {}
            meas_times_desc = {}
        
            sparsity_meas = {}
            sparsity_meas_descr = {} 
        
            meas_times_desc[1] = "Cov matrix multiplication"
            meas_times_desc[2] = "Cholmod: analyze 1"
            meas_times_desc[3] = "Cholmod: Cholesky in-place Ki"
            meas_times_desc[4] = "Cholmod: Cholesky in-place KiN"
            meas_times_desc[5] = "Derivative calculation part 1"
            meas_times_desc[6] = "Derivative calculation part 2"
            meas_times_desc[7] = "LML calculation main"            
            
            meas_times[7] = []
            meas_times_desc[8] = "Der p2: data fit term"
            
            meas_times[8] = []
            meas_times_desc[9] = "Der p2: determ deviratives (extra function)"
            
            meas_times[9] = []
            meas_times_desc[10] = "Der noise: part 1"
            
        
        HtH = np.dot(H.T, H)
        if measure_timings: t1 = time.time()
        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
        Ki = 0.5*(Ki + Ki.T)
        if measure_timings: meas_times[1] = time.time() - t1
        g_noise_var = float(g_noise_var)
        
        KiN = Ki +  GtG /g_noise_var# Precision with a noise
        if measure_timings: sparsity_meas[1] = Ki.getnnz()

        if measure_timings: t1 = time.time()
        analyzed_factor = cholmod.analyze(KiN) # perform only once this operation
                                              # since this is expensive?  
        if measure_timings: meas_times[2] = time.time() - t1        
        
        Ki_factor = analyzed_factor._clone()
        KiN_factor = analyzed_factor._clone()
        
        if measure_timings: t1 = time.time()
        # Ki_factor.cholesky_inplace(Ki) # There we error sometimes here, hence substitute to equivalent
        Ki_factor.cholesky_inplace(Qi)
        if measure_timings: meas_times[3] = time.time() - t1
            
        # Check 1 ->
#        Ko = np.linalg.inv(Ki.toarray())
#        K = G * Ko * G.T + np.eye(3) * g_noise_var
#        right_log_det = np.log( np.linalg.det(K) )
#        right_data_fit = np.dot(Y.T, np.dot( np.linalg.inv(K), Y ))
#        
#        p1_det = Ki_factor.logdet()
        # Check 1 <-
        
        # Check 3 ->
#        dKi1 = dKi_vector[0]
#        dKi2 = dKi_vector[1]
#        
#        Ki_inv = Ki_factor.inv()
#        d1 = - G * Ki_inv * dKi1 * Ki_inv * G.T # dK_dTheta
#        d2 = - G * Ki_inv * dKi2 * Ki_inv * G.T
#        
#        K_inv = np.linalg.inv(K)
#        right_det_deriv1 = -0.5*np.trace( np.dot(K_inv, d1.toarray() )) 
#            
#        right_det_deriv2 = -0.5*np.trace( np.dot(K_inv, d2.toarray() ))
#        
#        tt1 = np.dot(K_inv,Y)
#        right_data_fit_deriv1 = 0.5*np.dot( tt1.T, np.dot( d1.toarray(), tt1) )
#        right_data_fit_deriv2 = 0.5*np.dot( tt1.T, np.dot( d2.toarray(), tt1) )
#        
#        right_der1 = right_det_deriv1 + right_data_fit_deriv1
#        right_der2 = right_det_deriv2 + right_data_fit_deriv2
        # Check 3 <-
        if measure_timings: t1 = time.time()
        KiN_factor.cholesky_inplace(KiN, beta=0)        
        if measure_timings: meas_times[4] = time.time() - t1
            
        data_num = Y.shape[0] # number of data points
        deriv_number = len(dKi_vector) + 1 # plus Gaussian noise
        d_marginal_ll = np.zeros((deriv_number,1))        
        
        if measure_timings: t1 = time.time()
        if compute_derivatives:
            #deriv_factor = analyzed_factor._clone()
            for dd in range(0, deriv_number-1): #  the last parameter is Gaussian noise
            
                dKi = dKi_vector[dd]
#                if measure_timings: sparsity_meas[5].append(dKi.getnnz())
#                # First part of determinant
#                if measure_timings: t2 = time.time()
#                deriv_factor.cholesky_inplace(dKi, beta=0)
#                if measure_timings: meas_times[7].append(time.time() - t2)
#                    
#                (deriv_L, deriv_D) = deriv_factor.L_D()
#                deriv_L = deriv_L.tocsc()
#                deriv_D = deriv_D.tocsc()
#                
#                if measure_timings: t2 = time.time()
#                #L4 = Ki_factor.apply_P( deriv_factor.apply_Pt(deriv_L) )
#                L4=deriv_L
#                if measure_timings: meas_times[8].append(time.time() - t1)
#            
#                Ki_deriv_L.append(L4) # Check that there ara not need to do permutation
#                Ki_deriv_D.append(deriv_D)
                
#                if measure_timings: t2 = time.time()
#                L5 = Ki_factor.solve_L(L4) # Same size as KiN
#                if measure_timings: meas_times[9].append(time.time() - t2)
#                if measure_timings: sparsity_meas[2].append(L5.getnnz())
#                
#                if measure_timings: t2 = time.time()
#                dd1 = (Ki_factor.solve_D(L5)*deriv_D).multiply(L5).sum() * 0.5
#                if measure_timings: meas_times[10].append(time.time() - t2)
                    
                #ss1 = m1.sum()
#               d_marginal_ll[dd,1] = -0.5*(ss) 
            
                # Another way ->
                dd3 =  Kip.multiply(dKi.T).sum() * 0.5          
                # Another way <-                
                
                
                # Another way 2 ->
                #ss = Ki_factor.solve_A(dKi)            
                #dd2 = -0.5*( -ss.diagonal().sum() )
                # Another way 2 <-
                d_marginal_ll[dd,0] = dd3
        if measure_timings: meas_times[5] = time.time() - t1        
        # Check 2 ->        
        #p2_det = Ki_factor.logdet()
        #p3_det = (Y.size)*np.log( g_noise_var )
        
        #det2 = p2_det - p1_det + p3_det
        # Check 2 <- 
        
        # Contribution from det{ K^{-1} }
        if measure_timings: t1 = time.time()
        mll_log_det = -Ki_factor.logdet()# ignoring 0.5 and - sign.
        mll_log_det += KiN_factor.logdet() + (Y.size)*np.log(g_noise_var)
        if np.isnan(mll_log_det):
            raise ValueError("marginal ll: mll_log_det is None")
        
        KiNGtY = KiN_factor.solve_A(GtY)
        
        tmp1 = (GtY.T*KiNGtY).toarray()
        
        mll_data_fit_term = ( np.dot(Y.T,Y) /g_noise_var - tmp1/g_noise_var**2 ) # ignoring 0.5 and - sign.
        
        marginal_ll = -0.5 * ( mll_log_det + mll_data_fit_term + (Y.size)*log_2_pi)
        if measure_timings: meas_times[7] = time.time() - t1
            
        if measure_timings: t1 = time.time()
        d_deriv_true = []
        if compute_derivatives: 
            if measure_timings: t2 = time.time() # TODO maybe reimplement by stacking
            for dd in range(0, deriv_number-1): #  the last parameter is Gaussian noise (ignore it)
                dKi = dKi_vector[dd]
#                dL = Ki_deriv_L[dd]
#                dD = Ki_deriv_D[dd]
                
                # Another pard of determinant
#                ss = KiN_factor.solve_A( Ki_deriv_L[dd] )
#                ss = ss.power(2).sum()
#                d_marginal_ll[dd,1] += -0.5*(ss)
                
#                if measure_timings: t2 = time.time()
#                L5 = KiN_factor.solve_L( dL)
#                if measure_timings: meas_times[11].append(time.time() - t2)
#                if measure_timings: sparsity_meas[3].append(L5.getnnz())                
#                
#                if measure_timings: t2 = time.time()
#                dd1 = -(KiN_factor.solve_D(L5)*dD).multiply(L5).sum() * 0.5
#                if measure_timings: meas_times[12].append(time.time() - t2)
#                
#                
#                # Another way ->
                #ss = KiN_factor.solve_A(dKi)            
                #dd2 = -0.5*( ss.diagonal().sum() )
#                d_deriv_true.append( dd2 )
                # Another way <-
                #d_marginal_ll[dd,0] += dd1            
                
                # 
                
                d_marginal_ll[dd,0] += -0.5* (KiNGtY.T*(dKi*KiNGtY)).toarray()/g_noise_var**2 # Data fit term
            if measure_timings: meas_times[8].append(time.time() - t2)
            
            if measure_timings: t2 = time.time() # TODO maybe reimplement by stacking
            #d_determ, det_for_test = sparse_inference.second_deriv_determinant( data_num, 
            #            block_size, KiN, HtH, g_noise_var, dKi_vector, determ_derivatives)

# Use function deriv_determinant ->            
            (d_determ, det_for_test, tmp_inv_data) = sparse_inference.deriv_determinant( data_num, \
                                 block_size, HtH, g_noise_var, \
                                 matrix_blocks, None, compute_derivatives=True, deriv_num=deriv_number, \
                                 matrix_derivs=matrix_blocks_derivatives, compute_inv_main_diag=False)
# Use function deriv_determinant <-
                                 
# Use function deriv_determinant2 ->                                 
#            (d_determ, det_for_test, tmp_inv_data) = sparse_inference.deriv_determinant2( data_num, block_size, HtH, g_noise_var, 
#                                 matrix_blocks, None, matrix_blocks_derivatives, None,
#                                 compute_inv_diag=False, add_noise_deriv=True)
# Use function deriv_determinant2 <-                                 
                                 
            d_marginal_ll += -0.5* d_determ[:,np.newaxis]
            
            if measure_timings: meas_times[9].append(time.time() - t2)
                
            # Derivative wrt noise
            # Data term:
            if measure_timings: t2 = time.time()
            tmp2 = G*KiNGtY
            d_marginal_ll[-1,0] += -0.5* ( -1.0/g_noise_var**2 * np.sum( np.power(Y,2)) + \
                                            2.0/g_noise_var**3 * tmp1  - \
                                            1.0/g_noise_var**4 * (tmp2.T * tmp2) )
            if measure_timings: meas_times[10] = time.time() - t2
                
#            # Detarminant terms:
#            if measure_timings: t2 = time.time()
#            ss = KiN_factor.solve_A(GtG)  # TODO work on that
#            if measure_timings: meas_times[15] = time.time() - t2
#            if measure_timings: sparsity_meas[4] = ss.getnnz()
            
            #d_marginal_ll[-1,0] += -0.5*( -ss.diagonal().sum()/g_noise_var**2 + Y.size/g_noise_var )
            d_marginal_ll[-1,0] += -0.5*( Y.size/g_noise_var )
                
        if measure_timings: meas_times[6] = time.time() - t1
        
        
        if measure_timings:
            return marginal_ll, d_marginal_ll, meas_times, meas_times_desc, sparsity_meas, \
                sparsity_meas_descr
        else:
            return marginal_ll, d_marginal_ll
    
    @staticmethod
    def mean_var_calc(block_size, Y, Ait, Qi, GtY, G, GtG, H, g_noise_var, 
                    matrix_blocks, which_observed=None, inv_precomputed=None):
        """
        Input:        
        -----------------------
        block_size: int
            Size of the block
            
        Y: array(N,1)
            1D-array.  For test points there are zeros in corresponding positions.
            In the same positions where zeros are in which_observed.
        
        Ait, Qi, GtY, G, GtG, H: matrices
            Matrix data
        
        g_noise_var: float
            Noise variance
            
        matrix_blocks: dict
            Data with matrices info past directly into deriv_determinant.
        
        which_observed: None, or array(N,1) or array(1,N) or array(N,) 
            Array consisting of zeros and ones. Ones are in the position of
            training points (observed), zeros are in the position of 
            test points (not observed). If None, then all observed.
            Affects whether we add or not a diagonal.
            
        inv_precomputed: 
            What determinant computation function returns.
        
        Kip: sparce (not used anymore)
            Block diagonal matrix. The diagonal blocks are the same as in K.
            Required for ll derivarive computation.
        """
        g_noise_var = float(g_noise_var)
        HtH = np.dot(H.T, H)
        data_num = Y.shape[0]        
        #import pdb; pdb.set_trace()
        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
        Ki = 0.5*(Ki + Ki.T)
        
#        last_data_points = 0
#        diagg = np.ones((data_num,)); diagg[-last_data_points:] = 0
#        g_noise_var_to_det = diagg * g_noise_var; g_noise_var_to_det[-last_data_points:] = np.inf
#        GtG2 = sparse.kron( sparse.csc_matrix(np.diag(diagg)), sparse.csc_matrix( HtH ), format='csc')    
#        KiN = Ki +  GtG2 /g_noise_var# Precision with a noise
        if which_observed is not None:
            tmp1 = sparse.csc_matrix( G.T * sparse.csr_matrix( np.diag(np.squeeze(which_observed/g_noise_var)) ) )
            GtG = tmp1 * G
            
            GtNY = tmp1*sparse.csc_matrix(Y)
        
        else:
            GtG  = GtG/g_noise_var # Precision with a noise
            
            GtNY = GtY/g_noise_var
            
        KiN = Ki +  GtG
        
        analyzed_factor = cholmod.analyze(KiN) # perform only once this operation
                                              # since this is expensive?  
                
        KiN_factor = analyzed_factor._clone()
        KiN_factor.cholesky_inplace(KiN, beta=0)        
        
        KiNGtY = KiN_factor.solve_A(GtNY)
        
        if inv_precomputed is None:
            _, _, inv_precomputed = sparse_inference.deriv_determinant( data_num, \
                                 block_size, HtH, g_noise_var, \
                                 matrix_blocks, which_observed, compute_derivatives=False, deriv_num=None, \
                                 matrix_derivs=None, compute_inv_main_diag=True)
        
        #import pdb; pdb.set_trace()          
        # compute extra matrix which required in covariance ->
        Gt = sparse.csc_matrix( G.T )

#        # Get rid of off diagonal blocks ->
#        # mask to choose the block diagonal part
#        ones_mask = sparse.kron( sparse.eye( data_num, format='csc') ,sparse.csc_matrix( np.ones( (block_size, block_size) ) ), format='csc' )
#        tmp9 = ones_mask.multiply(Kip)
#        # Get rid of off diagonal blocks <-        
        
        #tmp12 = sparse.kron( sparse.csc_matrix( np.ones( (data_num,1) ) ), sparse.eye(block_size, format='csc'), format='csc')
        tmp12 = sparse.csc_matrix( np.ones( (data_num,1) ) )
        
        #cov_calc_rhs = (tmp11*tmp12).toarray()
        cov_calc_rhs = (Gt*tmp12).toarray()
        # compute extra matrix which required in covariance <-
        
#        # other (straightforward) result diag (test) ->
        #Ki_factor = analyzed_factor._clone()        
        #Ki_factor.cholesky_inplace(Ki, beta=0) 
        
#        # diagonal part of the variance
#        diag_var = (G*tmp10*sparse.csc_matrix( np.ones( (data_num,1) ) )).toarray()
#        
#        om2 = sparse.kron( sparse.eye( data_num, format='csc') ,sparse.csc_matrix( np.ones( (block_size, 1) ) ), format='csc' )
#        K = Ki_factor.solve_A( sparse.eye( data_num*block_size, format='csc') )        
#        cc_rhs = GtG * (K * Gt / g_noise_var)
#                
#        tt1 = KiN_factor.solve_A(cc_rhs)
#        other_res_diag = np.diag((G*tt1).toarray()); other_res_diag.shape = (other_res_diag.shape[0],1)
#        var = diag_var - other_res_diag
#        # other (straightforward) result diag (test) <-
        
        # Compute mean
        #mean_rhs = sparse.csc_matrix(Y/g_noise_var) - G*KiNGtY/g_noise_var # first term in this formula is incorrect. Need to account for inf. vars
        #mean = (G*Ki_factor.solve_A( Gt*mean_rhs)).toarray()
        
        mean2 = (G*KiNGtY).toarray()
        #import pdb; pdb.set_trace()   
        #Compute variance
        result_diag = sparse_inference.sparse_inv_rhs(data_num, block_size, 
                        matrix_blocks, HtH/g_noise_var , H, inv_precomputed, cov_calc_rhs,
                        which_observed) 
        
        #!!! HtH - H in sparse_inverse_cov and marginal_ll
        # One extra input in sparse_inv_rhs
                       
        return mean2, result_diag
    


    @staticmethod
    #@profile
    def deriv_determinant2( points_num, block_size, HtH, g_noise_var, 
                                 matrix_data, rhs_matrix_data, front_multiplier, which_observed=None,
                                 compute_inv_diag=False, add_noise_deriv=False):
        """
        This function is a different implementation of determinant term 
        (and its derivaives) in GP 
        marginal likelihood. It uses the formula d(log_det)/d (Theta) = Trace[ K d(K)\d(Theta)]
        Matrix K is assumed block tridiagonal, K^{-1} is sparse. Essentially what the function does:
        it compues the diagonal of (K d(K)\d(Theta) ). Having a diagonal it computes a trace. 
        
        Right now the summetricity assumption is used, but in general the method
        works for nonsymmetrix block tridiagonal matrices as well.        
        
        This function works for multiple rhs matrices so that all the dirivatives are computed at once!!!
        
        Notation for matrix is: K = A0, B1 0       
                                    C1, A1, B2
                                    0 , C2, A2
        Input:
        -----------------------------------
        
        points_num: int
            Number blocks
            
        block_size: int
            Size of the block            
        
        HtH: matrix (block_size, block_size)
            Constant matrix added to the diagonal blocks. It is divieded by g_noise_var
        
        g_noise_var: float
            U sed in connection with previous parameter for modify main diagonal blocks
            
        matrix_data: dictionary
            It is supposed to contain 2 keys: 'Ak' and 'Bk'. Each of those
            contain dictionary for main diagonal and upper-main diagonal.
            They are accessible by indices matrix_data['Ak'][0:points_num],
            matrix_data['Bk'][0:(points_num-1)]
            
        rhs_matrix_data: disctionary
            The same format as in "matrix_data".
            
        front_multiplier: (block_size, k). k- any matrix
            Matrix by which the whole expession K^{-1}D is multiplied in front.
            Actually, small block of this matrix.
        
         which_observed: None, or array(N,1) or array(1,N) or array(N,) 
            Array consisting of zeros and ones. Ones are in the position of
            training points (observed), zeros are in the position of 
            test points (not observed). If None, then all observed.
            Affects whether we add or not a diagonal.
            
        add_noise_deriv: bool
            Whether determinant noise derivative is computed. In this case, add one more derivative
                wrt noise. dAkd have the derivative HtH wrt noise. dCkd have zero.
        
        deriv_num: int
            Number of derivatives
        
        Output:
        -------------------------------------
        
        d_determinant: array(deriv_num,1) or None       
            Derivative of the determiant or None (if compute_derivatives == False)            
        
        determinant: float
            Determiant
        """
        
        Akd = matrix_data['Akd']
        Bkd = matrix_data['Bkd']
        
        rhs_Akd = rhs_matrix_data['dAkd']
        rhs_Ckd = rhs_matrix_data['dCkd']
         # Convert input matrices to 
        #HtH = HtH.toarray()
        inversion_comp={}; inversion_comp['d_l'] = {}; inversion_comp['d_r'] = {}
        inversion_comp['rhs_diag_d'] = {}; inversion_comp['rhs_diag_u'] = {}
        
        if which_observed is None:
            which_observed = np.ones( points_num )
        else:
            which_observed = np.squeeze(which_observed )
        
        HtH_zero = np.zeros( HtH.shape )
        extra_diag = lambda k: HtH if (which_observed[k] == 1) else HtH_zero
        
        prev_Lambda = Akd[0] + extra_diag(0)/g_noise_var #KiN[0:block_size, 0:block_size].toarray() #Akd[0] + HtH/g_noise_var
        prev_Lambda_back = Akd[ (points_num-1) ] + extra_diag((points_num-1))/g_noise_var
        
        determinant = 0
        d_determinant = None
        deriv_num = rhs_Akd[0].shape[0]
        if add_noise_deriv:
            deriv_num += 1
            
        d_determinant = np.zeros( (deriv_num,) )
            
        rhs_Ck = np.zeros( (deriv_num, block_size, block_size ) )
        rhs_Ak = np.zeros( (deriv_num, block_size, block_size ) )
        rhs_Ck_back = np.zeros( (deriv_num, block_size, block_size ) )
        rhs_Ak_back = np.zeros( (deriv_num, block_size, block_size ) )
        
        # In our notation the matrix consist of lower diagonal: C1, C1,...
        # main diagonal: A0, A1, A2..., and upper diagonal: B1, B2,...        
        # In this case the matrix is symetric.
        
        for i in range(0, points_num): # first point was for initialization
        
            (LL, cc) = la.cho_factor(prev_Lambda, lower = True)
            (LL1, cc1) = la.cho_factor(prev_Lambda_back, lower = True)
            
    
            inversion_comp['d_l'][i] = prev_Lambda 
            inversion_comp['d_r'][points_num-i-1] = prev_Lambda_back
                   
            determinant += 2*np.sum( np.log(np.diag(LL) ) ) # new
            
            # HELP 2 ->            
            # If we want to stack separate matrices in one dimansion
            # e. g. A(n,m,m) -> A(m, m*n), we use:
            # B = A.swapaxes(1,0).reshape(m,m*n)
            
            # If we want to transform back:
            # A = B.reshape(m,n,m).swapaxes(1,0)           

            if (i==points_num-1):
                break
                # Future points are not computed any more
            
            Bk = Bkd[i]# KiN[ind_start_lower:ind_end_lower, ind_start_higher:ind_end_higher].toarray() # Bkd[i]#
            Ak = Akd[i+1]+ extra_diag(i+1)/g_noise_var# KiN[ind_start_higher:ind_end_higher, ind_start_higher:ind_end_higher].toarray() #Akd[i+1]+ HtH/g_noise_var#
            
            Bk_back = Bkd[points_num-i-2] 
            Ak_back = Akd[points_num-i-2]+ extra_diag(points_num-i-2)/g_noise_var            
            if add_noise_deriv:
                rhs_Ck[0:-1,:,:] = rhs_Ckd[i];
                rhs_Ak[0:-1,:,:] = rhs_Akd[i+1]; rhs_Ak[-1,:,:] = extra_diag(i+1)
                
                rhs_Ck_back[0:-1,:,:] = rhs_Ckd[points_num-i-2]
                rhs_Ak_back[0:-1,:,:] = rhs_Akd[points_num-i-2]; rhs_Ak_back[-1,:,:] = extra_diag(points_num-i-2)
            else:
                rhs_Ck = rhs_Ckd[i]; 
                rhs_Ak = rhs_Akd[i+1]
                
                rhs_Ck_back = rhs_Ckd[points_num-i-2]
                rhs_Ak_back = rhs_Akd[points_num-i-2]

            # Compute rhs_Ak_tmp part ->
            tmp1 = np.transpose(rhs_Ck, (0,2,1) ) # first transpoce Ck to make it Bk
            tmp1 = tmp1.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # solve a system
            # Compute rhs_Ak_tmp part <-
            
            # Compute rhs_Ak_back_tmp part ->
            tmp2 = rhs_Ck_back.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # solve a system 
            # Compute rhs_Ak_back_tmp part <-
            
            # for doing every derivative simultaneously.
            rhs_Ak_tmp = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num)
            rhs_Ak_back_tmp = rhs_Ak_back.swapaxes(1,0).reshape(block_size,block_size*deriv_num)
            
            inversion_comp['rhs_diag_d'][i+1] = rhs_Ak_tmp - np.dot( Bk.T, la.cho_solve((LL, cc), tmp1 ) ) #tmp1
            inversion_comp['rhs_diag_u'][points_num-i-2] = rhs_Ak_back_tmp - np.dot( Bk, la.cho_solve((LL1, cc1), tmp2 ) ) 
            
            if i == 0: # incert necessary matrices for the first and last points
                rhs_Ak[0:-1,:,:] = rhs_Akd[0]; rhs_Ak[-1,:,:] = extra_diag(0)
                inversion_comp['rhs_diag_d'][0] = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # forward transform
                
                rhs_Ak[0:-1,:,:] = rhs_Akd[points_num-1]; rhs_Ak[-1,:,:] = extra_diag(points_num-1)
                inversion_comp['rhs_diag_u'][points_num-1] = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num)  # forward transform
        
            
            prev_Lambda_inv_term = la.cho_solve((LL, cc), Bk) # new            
            Lambda = Ak - np.dot(Bk.T, prev_Lambda_inv_term)
            prev_Lambda = Lambda # For the next step
        
            prev_Lambda_inv_term1 = la.cho_solve((LL1, cc1), Bk_back.T) # new 
            Lambda_back = Ak_back - np.dot( Bk_back, prev_Lambda_inv_term1)             
            prev_Lambda_back = Lambda_back
            del prev_Lambda_inv_term1, prev_Lambda_inv_term
        
        if front_multiplier is None:
            new_block_size = block_size
        else:
            new_block_size = front_multiplier.shape[0]
   
        Akd = matrix_data['Akd']
        d_l = inversion_comp['d_l']
        d_r = inversion_comp['d_r']
        d_d = inversion_comp['rhs_diag_d']
        d_u = inversion_comp['rhs_diag_u']        
        
        #result_diag = np.empty( (points_num*new_block_size, d_2 ) )
        #import pdb; pdb.set_trace()
        inv_diag = {}
        for i in range(0, points_num):
            #start_ind = block_size*i
            
            #lft = np.tile(np.eye( block_size), (deriv_num,) ) - d_d[i] - d_u[i]
            if add_noise_deriv:
                #rhs_Ck[0:-1,:,:] = rhs_Ckd[i];
                rhs_Ak[0:-1,:,:] = rhs_Akd[i]; rhs_Ak[-1,:,:] = extra_diag(i)
            else:
                rhs_Ak = rhs_Akd[i]
            tmp1 = rhs_Ak.swapaxes(1,0).reshape(block_size,block_size*deriv_num)
            
            lft =  -tmp1 + d_d[i] + d_u[i]            
            
            tmp = np.linalg.solve( -Akd[i] - extra_diag(i)/g_noise_var + d_l[i] + d_r[i], lft )
            
            tmp = tmp.reshape(block_size,deriv_num,block_size).swapaxes(1,0) # inverse transformation
            # Temporarily block the return of diagonal because we have several diagonals
            
            d_determinant += np.trace(tmp, axis1 = 1, axis2=2)
            inv_diag[i] = tmp[0, :, :]            
            
        if add_noise_deriv:
            d_determinant[-1] = -d_determinant[-1] / (g_noise_var**2)
                
        return d_determinant, determinant, inversion_comp

                                    
    @staticmethod
    #@profile
    def deriv_determinant( points_num, block_size, HtH, g_noise_var, 
                                 matrix_data, which_observed=None, compute_derivatives=False, deriv_num=None, 
                                 matrix_derivs=None, compute_inv_main_diag=False):
        """
        This function computes the log_detarminant,its derivatives: d_log_determinant,
        and 3 diagonals (not implemented) of the inverse of SYMMETRIC BLOCK TRIDIAGONAL MATRIX.
        
        It uses the method of differentiating the recursive formula for determinant.
        
        Right now the summetricity assumption is used, but in general the method
        works for nonsymmetrix block tridiagonal matrices as well.        
        
        Notation for matrix is: K = A0, B1 0       
                                    C1, A1, B2
                                    0 , C2, A2
        Input:
        -----------------------------------
        
        points_num: int
            Number blocks
            
        block_size: int
            Size of the block            
        
        HtH: matrix (block_size, block_size)
            Constant matrix added to the diagonal blocks. It is divieded by g_noise_var
        
        g_noise_var: float
            U sed in connection with previous parameter for modify main diagonal blocks
            
        matrix_data: dictionary
            It is supposed to contain 2 keys: 'Ak' and 'Bk'. Each of those
            contain dictionary for main diagonal and upper-main diagonal.
            They are accessible by indices matrix_data['Ak'][0:points_num],
            matrix_data['Bk'][0:(points_num-1)]
            
        which_observed: None, or array(N,1) or array(1,N) or array(N,) 
            Array consisting of zeros and ones. Ones are in the position of
            training points (observed), zeros are in the position of 
            test points (not observed). If None, then all observed.
            Affects whether we add or not a diagonal.
            
        compute_derivatives: bool
            Whether to compute determinant derivatives
        
        deriv_num: int
            Number of derivatives
            
        matrix_derivs: dictionary
            Derivatives of matrix blocks. With keys 'dAkd', 'dCkd'. Indices are
            similar to matrix_data parameter.
        
        compute_inv_main_diag: bool
            Whether to compute some intermidiate numbers do that later it is
            faster to compute the block diagonal of inverse of the matrix K.
            
        
        
        Output:
        -------------------------------------
        
        d_determinant: array(deriv_num,1) or None       
            Derivative of the determiant or None (if compute_derivatives == False)            
        
        determinant: float
            Determiant
            
        inversion_comp: dict{} 
            Dict with 2 keys 'd_l' and 'd_r'. There stored the information for computing
            inverse of tridiagonal matrix. If 'compute_inv_main_diag' the None.
        """
        
        Akd = matrix_data['Akd']
        Bkd = matrix_data['Bkd']
         # Convert input matrices to 
        #HtH = HtH.toarray()
        if isinstance(g_noise_var, np.ndarray):
            noise_vector = True
            noise_var = g_noise_var[0]
        else:
            noise_vector = False
            noise_var = g_noise_var
        
        if which_observed is None:
            which_observed = np.ones( points_num )
        else:
            which_observed = np.squeeze(which_observed )
        
        HtH_zero = np.zeros( HtH.shape )
        extra_diag = lambda k: HtH if (which_observed[k] == 1) else HtH_zero
        
        prev_Lambda = Akd[0] + extra_diag(0)/noise_var #KiN[0:block_size, 0:block_size].toarray() #Akd[0] + HtH/g_noise_var
        
        determinant = 0
        d_determinant = None
        if compute_derivatives:
            d_determinant = np.zeros( (deriv_num,) )
            
            dAkd = matrix_derivs['dAkd']
            dCkd = matrix_derivs['dCkd']
            
            dCk = np.zeros( (deriv_num, block_size, block_size ) )
            dAk = np.zeros( (deriv_num, block_size, block_size ) )
            prev_d_Lambda = np.empty( (deriv_num, block_size, block_size ) ) 
            #for j in range(0,deriv_num-1):
            #    prev_d_Lambda[j, :, :] = dKi_vector[j][0:block_size, 0:block_size].toarray()
            prev_d_Lambda[0:-1,:,:] = dAkd[0] # new
            prev_d_Lambda[-1, :, :] = extra_diag(0) # (-HtH/g_noise_var**2) # HtH # new
        
        inversion_comp = None
        if compute_inv_main_diag:
                
            inversion_comp={}; inversion_comp['d_l'] = {}; inversion_comp['d_r'] = {}
            prev_Lambda_back = Akd[ (points_num-1) ] + extra_diag((points_num-1))/noise_var
        
        # In our notation the matrix consist of lower diagonal: C1, C1,...
        # main diagonal: A0, A1, A2..., and upper diagonal: B1, B2,...        
        # In this case the matrix is symetric.
        
        for i in range(0, points_num): # first point was for initialization
            #import pdb; pdb.set_trace()
            (LL, cc) = la.cho_factor(prev_Lambda, lower = True)
            #KiN_ldet += np.log( la.det( prev_Lambda) ) # old         
            determinant += 2*np.sum( np.log(np.diag(LL) ) ) # new
                
            if compute_derivatives:
                # HELP 2 ->            
                # If we want to stack separate matrices in one dimansion
                # e. g. A(n,m,m) -> A(m, m*n), we use:
                # B = A.swapaxes(1,0).reshape(m,m*n)
                
                # If we want to transform back:
                # A = B.reshape(m,n,m).swapaxes(1,0)           
                
                # HELP 2 <-
                tmp = prev_d_Lambda.swapaxes(1,0).reshape(block_size,block_size*deriv_num) # solve a system 
                # for doing every derivative simultaneously.
                
                tmp = la.cho_solve((LL, cc), tmp )  # new          
                # tmp = la.solve(prev_Lambda, tmp) # old
                tmp = tmp.reshape(block_size,deriv_num,block_size).swapaxes(1,0) # inverse transformation
                
                d_determinant += np.trace( tmp, axis1 = 1, axis2 = 2)
            
            if compute_inv_main_diag:
                (LL1, cc1) = la.cho_factor(prev_Lambda_back, lower = True)
                
                inversion_comp['d_l'][i] = prev_Lambda 
                inversion_comp['d_r'][points_num-i-1] = prev_Lambda_back
                
            #print(i)
            if (i==points_num-1):
                break
                # Future points are not computed any more
            
#            ind_start_higher = (i+1)*block_size
#            ind_end_higher = ind_start_higher + block_size
#            ind_start_lower = i*block_size            
#            ind_end_lower = ind_start_lower + block_size
            if noise_vector:
                noise_var = g_noise_var[i+1]
                
            Bk = Bkd[i]# KiN[ind_start_lower:ind_end_lower, ind_start_higher:ind_end_higher].toarray() # Bkd[i]#
            Ak = Akd[i+1]+ extra_diag(i+1)/noise_var# KiN[ind_start_higher:ind_end_higher, ind_start_higher:ind_end_higher].toarray() #Akd[i+1]+ HtH/g_noise_var#
            
            #for j in range(0,deriv_num-1):
                
                #dCk[j, :, :] = dKi_vector[j][ind_start_higher:ind_end_higher, ind_start_lower:ind_end_lower].toarray()
                #dAk[j, :, :] = dKi_vector[j][ind_start_higher:ind_end_higher, ind_start_higher:ind_end_higher].toarray()
            
            prev_Lambda_inv_term = la.cho_solve((LL, cc), Bk) # new            
            Lambda = Ak - np.dot(Bk.T, prev_Lambda_inv_term)
            prev_Lambda = Lambda # For the next step
            
            if compute_inv_main_diag:
                Bk = Bkd[points_num-i-2] 
                Ak = Akd[points_num-i-2]+ extra_diag(points_num-i-2)/noise_var
            
                prev_Lambda_inv_term1 = la.cho_solve((LL1, cc1), Bk.T) # new 
            
                Lambda_back = Ak - np.dot( Bk, prev_Lambda_inv_term1)             
            
                prev_Lambda_back = Lambda_back
                del prev_Lambda_inv_term1
                
            if compute_derivatives:
                dCk[0:-1, :, :] = dCkd[i] # new
                dAk[0:-1, :, :] = dAkd[i+1] # new
                dAk[-1, :, :] = extra_diag(i+1) # -HtH/g_noise_var**2 # new # HtH
            
                # HELP 1 ->
                # If we have 3d-array A(n, m, m) and want to multiply by B(m, m) n times:
                # e.g ( A[0] * B, A[1] * B,  ...) then we do:
                # C = np.dot(A,B)
                
                # The same but with B.T:
                # C = np.dot(A,B.T)
                
                # If we want to compute: B(m, m)* A(n ,m ,m) we do:
                # C = np.transpose(np.dot(np.transpose(A,(0,2,1)),B.T),(0,2,1))
                
                # the same but: B.T * A
                # C = np.transpose(np.dot(np.transpose(A,(0,2,1)),B),(0,2,1))            
                
                # ! Everywhere insted of np.dot we can use np.multiply !
                # HELP 1 <-
                d1 = np.dot( dCk, prev_Lambda_inv_term)
                d2 = np.dot( prev_d_Lambda, prev_Lambda_inv_term)
                d2 = np.transpose(np.dot(np.transpose(d2,(0,2,1)),prev_Lambda_inv_term),(0,2,1))
                
                d_Lambda = dAk - d1 - np.transpose(d1, (0,2,1)) + d2
                prev_d_Lambda = d_Lambda # For the next step
        
        if compute_derivatives:
            if noise_vector:
                raise NotImplemented
            d_determinant[-1] = -d_determinant[-1] / (g_noise_var**2)
                
        return d_determinant, determinant, inversion_comp
    
    @staticmethod    
    def sparse_inv_rhs(points_num, block_size, matrix_data, 
                       extra_matrix_block_diag, front_multiplier, 
                       inversion_comp, rhs, which_observed=None):
        """
        Function computes diagonal blocks of the (inverse tri-diag times matrix)
        given some precalculated data in the function 'deriv_determinant' and
        passed in 'inversion_comp'.
        
        Inputs:
        -----------------------
        points_num:
        
        block_size:
        
        matrix_data: dictionary
            It is supposed to contain 2 keys: 'Ak' and 'Bk'. Each of those
            contain dictionary for main diagonal and upper-main diagonal.
            They are accessible by indices matrix_data['Ak'][0:points_num],
            matrix_data['Bk'][0:(points_num-1)]
            
        extra_matrix_block_diag: matrix(block_size, block_size)
            On each iteration this matrix is added to the diagonal block.
            It is typically HtH/g_noise_var.
            
        front_multiplier: matrix(*, block_size)
            On each iteration this matrix is multiplied by the result of the inversion
            
        rhs: matrix(block_size, block_size) or matrix(block_size*points_num, block_size)
            If it is a larger matrix then blocks are in given in a column.
            If it smaller matrix then it is assumed that all blocks are the same and only one is given.
        
        which_observed: None, or array(N,1) or array(1,N) or array(N,) 
            Array consisting of zeros and ones. Ones are in the position of
            training points (observed), zeros are in the position of 
            test points (not observed). If None, then all observed.
            Affects whether we add or not a diagonal.
            
        Output:
        -----------------------
        """
        (d_1, d_2) = rhs.shape
        
        if (d_1 == block_size):
            rhs_small=True
        elif (d_1 == block_size*points_num):
            rhs_small=False
        else:
            raise ValueError("sparse_inv_rhs: Incorrect rhs.")
        
        if which_observed is None:
            which_observed = np.ones( points_num )
        else:
            which_observed = np.squeeze(which_observed )
        
        HtH_zero = np.zeros( extra_matrix_block_diag.shape )
        extra_diag = lambda k: extra_matrix_block_diag if (which_observed[k] == 1) else HtH_zero
   
        if front_multiplier is None:
            new_block_size = block_size
        else:
            new_block_size = front_multiplier.shape[0]
   
        Akd = matrix_data['Akd']
        d_l = inversion_comp['d_l']
        d_r = inversion_comp['d_r']
        
        result_diag = np.empty( (points_num*new_block_size, d_2 ) )
        for i in range(0, points_num):
            start_ind = block_size*i
            if rhs_small:
                lft = rhs
            else:
                lft = rhs[start_ind:start_ind+block_size, :]

            tmp = np.linalg.solve( -Akd[i] - extra_diag(i) + d_l[i] + d_r[i], lft )
            
            start_ind2 = new_block_size*i
            if front_multiplier is None:
                result_diag[start_ind2:(start_ind2+new_block_size),:] = tmp
            else:
                result_diag[start_ind2:(start_ind2+new_block_size),:] = np.dot(front_multiplier, tmp)
            
        return result_diag
    
    @staticmethod 
    def covariances():
        """
        This function computes variances of data.
        
        Input:
        -----------------------
        
        
        """


        