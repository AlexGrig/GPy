# -*- coding: utf-8 -*-
# Copyright (c) 2016, Alex Grigorevskiy
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Testing functions for sparse precision 1-D Gaussian processes.
"""

import unittest
import numpy as np
import scipy as sp
import GPy
import sksparse.cholmod as cholmod
import scipy.sparse as sparse
import scipy.linalg as la
#import GPy.models.state_space_model as SS_model
#from .state_space_main_tests import generate_x_points, generate_sine_data, \
#    generate_linear_data, generate_brownian_data, generate_linear_plus_sin

import time
import GPy.models.state_space_model as ss_model
import GPy.models.ss_sparse_model as ss_sparse_model

from nose import SkipTest

#from GPy.inference.latent_function_inference.ss_sparse_inference import sparse_inference
from GPy.inference.latent_function_inference.ss_sparse_inference import btd_inference
#from GPy.inference.latent_function_inference.ss_sparse_inference import SparsePrecision1DInference

def generate_data(n_points,x_lower_value=0.0, x_upper_value=200.0):
    """
    Input:
    -----------
    
    n_points: number of data points.
    """

    x_points = np.linspace(x_lower_value, x_upper_value, n_points)

    # The signal is a sum of 2 sinusoids + noise
    
    noise_variance = 0.5
    # 1-st sunusoid
    magnitude_1 = 2.0
    phase_1 = 0.5
    period_1 = 23
    
    # 1-st sunusoid
    magnitude_2 = 5.0
    phase_2 = 0.1
    period_2 = 5
    
    y_points = magnitude_1*np.sin( x_points* 2*np.pi/ period_1 + phase_1) + \
                magnitude_2*np.sin( x_points* 2*np.pi/ period_2 + phase_2)
                
                
    y_points = y_points + noise_variance * np.random.randn(n_points,)            
    
    x_points.shape = (n_points,1)    
    y_points.shape = (n_points,1) 
    
    return x_points, y_points


def plot(x_data, y_data, mean, gp_mean, var, gp_var, other_var):
    """
    Plot Results
    """    
    
    import matplotlib.pyplot as plt
    
    plt.figure(1)
    plt.plot(x_data, y_data ,'.k', label = 'data')    
    plt.plot(x_data, mean, '-r', label='sparse')
    plt.plot(x_data, gp_mean, '-b', label = 'orig-gp')
    
    # Vars
    plt.plot(x_data, mean + var, '--r', label='sparse')
    plt.plot(x_data, mean - var, '--r', label='sparse')
    plt.plot(x_data, gp_mean + gp_var, '--b', label = 'orig-gp')    
    plt.plot(x_data, gp_mean - gp_var, '--b', label = 'orig-gp')
    
    plt.plot(x_data, gp_mean + other_var, '--m', label = 'other-gp') 
    plt.plot(x_data, gp_mean - other_var, '--m', label = 'other-gp') 
    #plt.plot(x_data, other_mean, '-m', label = 'orig-gp')
    
    plt.show()

class ComputationalRoutinesTests(np.testing.TestCase):
    """
    Test basic computational routines  
    
    """
    
    def setUp(self):
        pass
    
    def test_reg_system_solution(self,):
        """
        Tests function block_tridiag_solver function. In particular tests
        usual system solution and determinant computation.
        """
        ii = np.random.randint(0,2)
        np.random.seed(234 + ii)
        
        N = 100 # number of blocks
        K = 3 # block size
        rhs_num = 6 # rhs number


        A = np.zeros( (N*K, N*K))
        A_D = np.empty((N*K, K))
        A_C = np.empty(((N-1)*K, K))
        rhs = np.empty((N*K, rhs_num))
        # generate matrices
        #import pdb; pdb.set_trace()
        for i in range(0,N):
            curr_A_D = np.random.rand(K,K)
            curr_A_C = np.random.rand(K,K)


            A[ (i+0)*K:(i+1)*K, (i+0)*K:(i+1)*K] = curr_A_D
            A_D[(i+0)*K:(i+1)*K,:] = curr_A_D            
            
            if i != (N-1):
                A_C[(i+0)*K:(i+1)*K,:] = curr_A_C
                
                A[ (i+1)*K:(i+2)*K, (i+0)*K:(i+1)*K] = curr_A_C
                A[ (i+0)*K:(i+1)*K, (i+1)*K:(i+2)*K] = curr_A_C.T
            
            rhs[(i+0)*K:(i+1)*K,:] = np.random.rand(K,rhs_num)
        
        #import pdb; pdb.set_trace()
        # function call
        (r1,r3,r4,inv_comp) = btd_inference.block_tridiag_solver(N,K, A_D, A_C, rhs, rhs_diag=False, inversion_comp=None, 
                                rhs_C_matr=None, comp_matrix_determinant=True, comp_solution=True, 
                                comp_solution_trace=False, rhs_block_num=None, rhs_block_width = None, front_multiplier=None)
                                
#            block_tridiag_solver(N,K, A_matr, C_matr, rhs_D_matr, rhs_diag=False, inversion_comp=None, rhs_C_matr=None,
#         comp_matrix_determinant=False, comp_solution=False, comp_solution_trace=False,
#         rhs_block_num=None, rhs_block_width = None, front_multiplier=None)
                                
        # determinant, solution, trace, inversion_comp

        true_sol = la.solve( A, rhs)
        true_det_sign = np.sign( la.det(A) )
        true_logdet  = np.log( np.abs( la.det(A)) )

        #import pdb; pdb.set_trace()
        
        tt1 = np.max( np.abs( true_sol - r3) )
        tt3 = np.max( np.abs( true_logdet - r1) )
        
        np.testing.assert_array_almost_equal(r3, true_sol, decimal=9)
        np.testing.assert_array_almost_equal(r1, true_logdet, decimal=12)
        #np.testing.assert_equal( tt2, True)
        return inv_comp
        
    def runBTDsolution(self,N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr=False,
                       front_multiplier=None):
        """
        Main function for testing block_tridiag_solver function. In particular tests
        BTD system solution (inverse of solution) and traces are computed.
        
        Probably only the saving "inversion_comp" functionality is not tested yet.
        """
        
#        N = 3 # number of blocks
#        K = 2 # block size
#        rhs_block_num = 1 # number of rhs blocks of K in one block

        A = np.zeros( (N*K, N*K))
        A_D = np.empty((N*K, K))
        A_C = np.empty(((N-1)*K, K))
        
        
        rhs_D = np.empty(( N*K, rhs_block_num*rhs_block_width ))
        rhs_C = np.empty(( (N-1)*K, rhs_block_num*rhs_block_width ))
        rhs_matr = np.zeros(( N*K, rhs_block_num*rhs_block_width*N ))
        
        # generate matrices
        #import pdb; pdb.set_trace()
        for i in range(0,N):
            curr_A_D = np.random.rand(K,K); curr_A_D = 0.5*(curr_A_D + curr_A_D.T)
            curr_A_C = np.random.rand(K,K)

            A[ (i+0)*K:(i+1)*K, (i+0)*K:(i+1)*K] = curr_A_D
            A_D[(i+0)*K:(i+1)*K,:] = curr_A_D            
            
            curr_rhs_D = np.random.rand(K, rhs_block_num*rhs_block_width ); #curr_rhs_D = 0.5*(curr_rhs_D + sparse_inference.transpose_submatrices(curr_rhs_D,rhs_block_width,rhs_block_num))
            if no_rhs_C_matr:
                curr_rhs_C = np.zeros( (K, rhs_block_num*rhs_block_width) )
            else:
                curr_rhs_C = np.random.rand(K, rhs_block_num*rhs_block_width )            
            
            rhs_D[(i+0)*K:(i+1)*K,:] = curr_rhs_D
            rhs_matr[(i+0)*K:(i+1)*K, (i+0)*rhs_block_width*rhs_block_num:(i+1)*rhs_block_width*rhs_block_num ] = curr_rhs_D
            
            if i != (N-1):
                A_C[(i+0)*K:(i+1)*K,:] = curr_A_C
                
                A[ (i+1)*K:(i+2)*K, (i+0)*K:(i+1)*K] = curr_A_C
                A[ (i+0)*K:(i+1)*K, (i+1)*K:(i+2)*K] = curr_A_C.T
                
                rhs_C[(i+0)*K:(i+1)*K,:] = curr_rhs_C
                
                if not no_rhs_C_matr:
                    rhs_matr[(i+1)*K:(i+2)*K, (i+0)*K*rhs_block_num:(i+1)*K*rhs_block_num ] = curr_rhs_C
                    rhs_matr[(i+0)*K:(i+1)*K, (i+1)*K*rhs_block_num:(i+2)*K*rhs_block_num ] = btd_inference.transpose_submatrices(curr_rhs_C,rhs_block_num, rhs_block_width) 
            
        #import pdb; pdb.set_trace()
        # function call
        (r1,r2,r3,inv_comp) = btd_inference.block_tridiag_solver(N,K, A_D, A_C, rhs_D, rhs_diag=True, inversion_comp=None, 
                                rhs_C_matr= (rhs_C if no_rhs_C_matr==False else None),
                                 comp_matrix_determinant=True, comp_solution=True, 
                                 comp_solution_trace= (K==rhs_block_width) and (front_multiplier is None),
                                 rhs_block_num=rhs_block_num, rhs_block_width = rhs_block_width, front_multiplier=front_multiplier)
                             
        # determinant, solution, trace, inversion_comp
        #import pdb; pdb.set_trace()
        true_sol_full = la.solve( A, rhs_matr)
        if front_multiplier is not None:
            front_size = front_multiplier.shape[0]
            true_sol_full = np.dot( np.kron( np.eye(N), front_multiplier) ,true_sol_full)
        else:
            front_size = K
            
        
        true_trace = np.zeros( (rhs_block_num,) )
        true_sol = np.empty( ( N*front_size, rhs_block_num*rhs_block_width ) )
        
        for i in range(0,N):
            ss_tmp = true_sol_full[(i+0)*front_size:(i+1)*front_size, (i+0)*rhs_block_width*rhs_block_num:(i+1)*rhs_block_width*rhs_block_num ]
            true_sol[(i+0)*front_size:(i+1)*front_size,:] = ss_tmp.copy()
            
            if (rhs_block_width == K) and (front_size == K): # Trace makes sence
                ff_tmp = ss_tmp.reshape(K,rhs_block_num,K).swapaxes(1,0)
                true_trace += ff_tmp.trace(axis1=1,axis2=2)
            
        true_logdet  = np.log( np.abs( la.det(A)) )

        #import pdb; pdb.set_trace()
        
        tt1 = np.max( np.abs( true_sol - r2) )
        tt3 = np.max( np.abs( true_logdet - r1) )
        
        tt4 = np.max( np.abs(r3- true_trace) ) if ((rhs_block_width == K) and  (front_multiplier is None)) else None       
        
        np.testing.assert_array_almost_equal(tt1, 0, decimal=8)
        np.testing.assert_array_almost_equal(tt3, 0, decimal=11)
        
        if tt4 is not None:
            np.testing.assert_array_almost_equal(tt4, 0, decimal=8)
        pass
    
        return inv_comp
        
    def test_btd_system_solution1(self,):
        """
        Tests function block_tridiag_solver function. 
        
        Test Square RHS blocks, several RHS.
        """
        ii = np.random.randint(0,2)
        np.random.seed(234 + ii)
        
        # Test regular case
        N = 100
        K= 3
        rhs_block_width = 3
        rhs_block_num = 1
        no_rhs_C_matr=False
        front_multiplier = None
        
        self.runBTDsolution(N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr,
                       front_multiplier)
        
        # Test multiple RHSs:
        rhs_block_num = 4
        self.runBTDsolution(N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr,
                       front_multiplier)

        # Test no_rhs_C_matr=True
        no_rhs_C_matr=True
        self.runBTDsolution(N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr,
                       front_multiplier)
        
        # Test fron multiplier:
        front_multiplier = np.random.rand(1,3)
        self.runBTDsolution(N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr,
                       front_multiplier)

    def test_btd_system_solution2(self,):
        """
        Tests function block_tridiag_solver function. 
        
        Test NON-SQUARE RHS blocks, several RHS.
        """
        ii = np.random.randint(0,2)
        np.random.seed(234 + ii)
        
         # Test regular case
        N = 100
        K= 3
        rhs_block_width = 2
        rhs_block_num = 1
        no_rhs_C_matr=True # Non-square RHS blocks works only with no subdiagonal blocks
        front_multiplier = None
        
        self.runBTDsolution(N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr,
                       front_multiplier)
        
        # Test multiple RHSs:
        rhs_block_num = 4
        self.runBTDsolution(N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr,
                       front_multiplier)
       
        # Test fron multiplier:
        front_multiplier = np.random.rand(1,3)
        self.runBTDsolution(N,K,rhs_block_width,rhs_block_num, no_rhs_C_matr,
                       front_multiplier)
                       
    def runTransposeSubmatrices(self,N, S, K):
        """
        The function tests the function btd_inference transpose submatrices.
        Transpose is only valid for square matrices.        
        
        Input:
        -------------------
        
        N: int
            Number of submatrices
        S: int
            Stacked submatrix width (row=True), or height (row=False)
        K: int
           Other submatrix dimension.
        """        
        
        M_row_wise = np.empty( (K, N*S) )
        M_row_wise_transposed = np.empty( (S, N*K) )
        M_column_wise = np.empty( (N*S, K) )
        M_column_wise_transposed = np.empty( (N*K, S) )
        
        #import pdb;pdb.set_trace()
        
        for i in range(0,N):
            
            curr_submatr = np.random.rand(K, S); 
                       
            M_row_wise[:, (i+0)*S:(i+1)*S] = curr_submatr
            M_row_wise_transposed[:, (i+0)*K:(i+1)*K] = curr_submatr.T
            
            M_column_wise[(i+0)*S:(i+1)*S,:] = curr_submatr.T
            M_column_wise_transposed[(i+0)*K:(i+1)*K,:] = curr_submatr
        
        #import pdb;pdb.set_trace()
        
        res_row_wise = btd_inference.transpose_submatrices(M_row_wise,N,S, row=True)        
        res_column_wise = btd_inference.transpose_submatrices(M_column_wise,N,S, row=False)
        
        np.testing.assert_array_almost_equal(res_row_wise, M_row_wise_transposed, decimal=16)
        np.testing.assert_array_almost_equal(res_column_wise, M_column_wise_transposed, decimal=16)
    
    def test_TransposeSubmatrices(self,):
        """
        
        """
        
        N = 3
        K = 3
        S=2
        self.runTransposeSubmatrices(N, S, K)
    
    def runBTDmultiplication(self, N, K, grad_params_no=1, no_subdiag=False, Kv=None):
        """
        The function tests multiplication of SYMMETRIC!(Symmetricity only wrt subdiagonals) BTD matrix by a vector.
        Actually several BTD matrices can be multiplied by a vector simultaneously.
        In this case these matrices are stacked together along the 1-st axis.
        E.g D[K, K*grad_params_no] where grad_params_no is the number of matrices.
        
        Input:
        ----------------
        
        N: int
          Number of blocks (or datapoints)
        K: width of the block
        
        grad_params_no: int
            Number of matrices stacked together.
        
        no_subdiag: bool
            If true the the matrix is assumed to be block-diagonal
            
        Kv: int
            Vertical block size. It might be different from block size K when
            there are no matr_low_D.
        """
        
        if Kv is None:        
            Kv = K
            
        # Generate matrix(ces) and vector ->
        D = np.empty(( N*Kv, K*grad_params_no )) # diagonal blocks
        C = np.empty(( (N-1)*K, K*grad_params_no )) # lower diagonal blocks
            
        bulk_true_matrix = np.zeros( (grad_params_no, N*Kv, N*K) )
        # generate matrices
        #import pdb; pdb.set_trace()
        for i in range(0,N):
            
            
            curr_D = np.random.rand(Kv, K*grad_params_no ); #curr_D = 0.5*(curr_D + btd_inference.transpose_submatrices( curr_D,grad_params_no,K))
            D[(i+0)*Kv:(i+1)*Kv,:] = curr_D
            
            if not no_subdiag:
                curr_C = np.random.rand(K, K*grad_params_no )            
                if i < (N-1):
                    C[(i+0)*K:(i+1)*K, :] = curr_C
            
            for j in range(0,grad_params_no):
                bulk_true_matrix[j,(i+0)*Kv:(i+1)*Kv,(i+0)*K:(i+1)*K] = curr_D[:, (j+0)*K:(j+1)*K ]
                
                if not no_subdiag:
                    if i < (N-1):
                        bulk_true_matrix[j,(i+1)*K:(i+2)*K,(i+0)*K:(i+1)*K] = curr_C[:, (j+0)*K:(j+1)*K ]
                        bulk_true_matrix[j,(i+0)*K:(i+1)*K,(i+1)*K:(i+2)*K] = curr_C[:, (j+0)*K:(j+1)*K ].T
                    
                    
        vect = np.random.rand( N*K, 1)
        # Generate matrix(ces) and vector <-
        
        #import pdb; pdb.set_trace()
        # Multiplication ->
        res = btd_inference.btd_times_vector(D, vect, matr_low_D=(C if no_subdiag is False else None), grad_params_no=grad_params_no, Kv=Kv)
        # Multiplication <-
   
        #import pdb; pdb.set_trace()
        # Verify the result ->
        for j in range(0,grad_params_no):
            res_true = np.dot( bulk_true_matrix[j,:,:], vect )
            tt = np.max( np.abs( res[:,j] - res_true))
            np.testing.assert_array_almost_equal(res_true.squeeze(), res[:,j], decimal=12)
        # Verify the result <-
    
    def test_BTD_multiplication(self,):
        """
        
        """
        ii = np.random.randint(0,2)
        np.random.seed(234 + ii)       
        
        N = 100
        K = 13
        grad_params_no = 1
        no_subdiag=False
        
        self.runBTDmultiplication(N, K, grad_params_no, no_subdiag)
        
        N = 100
        K = 7
        grad_params_no = 1
        no_subdiag=True # No subdiag
        
        self.runBTDmultiplication(N, K, grad_params_no, no_subdiag)
        
        
        N = 100
        K = 9
        grad_params_no = 7 # Nonzero derivative number
        no_subdiag=False
        
        self.runBTDmultiplication(N, K, grad_params_no, no_subdiag)
        
        N = 100
        K = 9
        Kv = 4
        grad_params_no = 7 # Nonzero derivative number
        no_subdiag=True
        
        self.runBTDmultiplication(N, K, grad_params_no, no_subdiag,Kv)
        
class SparsePrecitionMLLTests(np.testing.TestCase):
    """
    The file tests marginal likelihood (MLL) and its derivatives calculation.    
    
    """
    
    def setUp(self):
        pass


#    def run_with_reg_GP3(self, x_data, y_data, kernel1, kernel2, noise_var,
#                        mll_compare_decimal=2, d_mll_compare_decimal=2):
#        """
#        The main function which is run for different settings.
#        """
#
#        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
#        block_size = F.shape[1]    
#        
#        grad_calc_params = {}
#        grad_calc_params['dP_inf'] = dP_inft
#        grad_calc_params['dF'] = dFt
#        grad_calc_params['dQc'] = dQct
#        
#        # Do not change with repetiotions
#        
#        mll_diff = None
#        d_mll_diff = None
#        #d_mll_diff_relative = None
#        
#        run_sparse = True
#        run_gp = True
#        if run_sparse:
#            #print('Sparse GP run:')
#            (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
#             matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(x_data, 
#                    y_data, F, L, Qc, P_inf, P0, H, p_largest_cond_num=1e20, compute_derivatives=True,
#                               grad_calc_params=grad_calc_params, p_regularization_type=2)
#            
#                               
#            res = sparse_inference.marginal_ll( block_size, y_data, Ait, Qi, \
#            GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, 
#                Kip=Kip, matrix_blocks= matrix_blocks, 
#                matrix_blocks_derivatives = matrix_blocks_derivatives)
#            
#            marginal_ll = res[0]; d_marginal_ll = res[1]
#
#        if run_gp:
#            #print('Regular GP run:')
#            gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
#            
#        mll_diff = marginal_ll - (-gp_reg.objective_function())
#        reg_d_mll = -gp_reg.objective_function_gradients(); reg_d_mll.shape = (reg_d_mll.shape[0],1)
#        d_mll_diff = d_marginal_ll - (reg_d_mll)
#        #d_mll_diff_relative = np.sum( np.abs(d_mll_diff) ) / np.sum( np.abs(reg_d_mll) ) 
#        import pdb; pdb.set_trace()
#        
#        np.testing.assert_array_almost_equal(marginal_ll, -gp_reg.objective_function(), mll_compare_decimal)
#        np.testing.assert_array_almost_equal(d_marginal_ll, reg_d_mll, d_mll_compare_decimal)


    def run_with_reg_GP1(self, x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal, d_mll_compare_decimal, p_regularization_type=2, p_largest_cond_num=1e13):
        """
        Tests that the function "test_marginal_ll" works properly.
        This is a test for test function. Compares computations with 
        GPy Gaussian Process regression. The idea is that formulas in
        "test_marginal_ll" are straight forward and simpler than in
        "marginal_ll".
        """
        
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
        block_size = F.shape[1]   
        N = y_data.shape[0]
        
        run_sparse = True
        run_gp = True
        
        #import pdb; pdb.set_trace()
        
        if run_sparse:
            _, _, Ki_logdet, _, _, d_Ki_logdet, Ki, dKi = btd_inference.test_build_matrices(x_data, y_data, 
                               F, L, Qc, P_inf, H, p_largest_cond_num, p_regularization_type, 
                               compute_derivatives=True, dP_inf=dP_inft, dF=dFt, dQc=dQct)
        
            #import pdb; pdb.set_trace()
            marginal_ll, d_marginal_ll, mll_data_fit_term, mll_log_det, mll_data_fit_deriv, mll_determ_deriv = btd_inference.test_marginal_ll(y_data, Ki, Ki_logdet, H, noise_var, 
                    compute_derivatives=True, d_Ki=dKi, d_Ki_logdet = d_Ki_logdet)
        
        #import pdb; pdb.set_trace()
        if run_gp:
            gp_reg = GPy.models.GPRegression(x_data,y_data, kernel2, noise_var=noise_var)
            
            K = gp_reg.posterior._K + np.eye(N)*noise_var

            mll_data_fit_term_true = np.dot( y_data.T, np.linalg.solve(K,y_data) )
            _, mll_log_det_true = np.linalg.slogdet(K)
            
            marginal_ll_true = -0.5*(mll_data_fit_term_true + mll_log_det_true +y_data.size*np.log(2*np.pi) )
            
            marginal_ll_true2 = (-gp_reg.objective_function())
            
            d_marginal_ll_true2 = -gp_reg.objective_function_gradients(); d_marginal_ll_true2.shape = (d_marginal_ll_true2.shape[0],1)
        
        #import pdb; pdb.set_trace()
        
        np.testing.assert_array_almost_equal(mll_data_fit_term, mll_data_fit_term_true, mll_compare_decimal)
        np.testing.assert_array_almost_equal(mll_log_det, mll_log_det_true, mll_compare_decimal)
        np.testing.assert_array_almost_equal(marginal_ll, marginal_ll_true, mll_compare_decimal)
        np.testing.assert_array_almost_equal(marginal_ll, marginal_ll_true2, mll_compare_decimal)
        
        np.testing.assert_array_almost_equal(d_marginal_ll, d_marginal_ll_true2, d_mll_compare_decimal)
    
    def run_with_reg_GP2(self, x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal, d_mll_compare_decimal, p_regularization_type=2, p_largest_cond_num=1e13):
        """
        Tests that the function "marginal_ll" works properly.
        Compares computations with "test_marginal_ll" and "marginal_ll"
        functions.
        """
        
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
        block_size = F.shape[1]   
        
        run_sparse_test = True
        run_sparse_btd = True
        
        #import pdb; pdb.set_trace()
        
        
        if run_sparse_test:
            _, _, Ki_logdet_true, _, _, d_Ki_logdet_true, Ki, dKi = btd_inference.test_build_matrices(x_data, y_data, 
                               F, L, Qc, P_inf, H, p_largest_cond_num, p_regularization_type, 
                               compute_derivatives=True, dP_inf=dP_inft, dF=dFt, dQc=dQct)
        
            #import pdb; pdb.set_trace()
            marginal_ll_true, d_marginal_ll_true, mll_data_fit_term_true, mll_log_det_true, \
            mll_data_fit_deriv_true, mll_determ_deriv_true = btd_inference.test_marginal_ll(y_data, Ki, Ki_logdet_true, H, noise_var, 
                    compute_derivatives=True, d_Ki=dKi, d_Ki_logdet = d_Ki_logdet_true)
        
        #import pdb; pdb.set_trace()
        if run_sparse_btd:
            
            Ki_diag, Ki_low_diag, Ki_logdet, d_Ki_diag, d_Ki_low_diag, d_Ki_logdet = btd_inference.build_matrices(x_data, y_data, 
                        F, L, Qc, P_inf, P0, H, p_largest_cond_num, p_regularization_type, 
                        compute_derivatives=True, dP_inf= dP_inft, dP0 = dP0t, dF=dFt, dQc= dQct)
            
            #import pdb; pdb.set_trace()
            marginal_ll, d_marginal_ll, mll_data_fit_term, mll_log_det, mll_data_fit_deriv, mll_determ_deriv = btd_inference.marginal_ll(block_size, 
                    y_data, Ki_diag, Ki_low_diag, Ki_logdet, H, noise_var, 
                    compute_derivatives=True, d_Ki_diag=d_Ki_diag, d_Ki_low_diag=d_Ki_low_diag, d_Ki_logdet = d_Ki_logdet)
        
        #import pdb; pdb.set_trace()
        
        np.testing.assert_array_almost_equal(mll_data_fit_term, mll_data_fit_term_true, mll_compare_decimal)
        np.testing.assert_array_almost_equal(mll_log_det, mll_log_det_true, mll_compare_decimal)
        np.testing.assert_array_almost_equal(marginal_ll, marginal_ll_true, mll_compare_decimal)
        
        np.testing.assert_array_almost_equal(mll_data_fit_deriv, mll_data_fit_deriv_true, d_mll_compare_decimal)
        np.testing.assert_array_almost_equal(mll_determ_deriv, mll_determ_deriv_true, d_mll_compare_decimal)
        np.testing.assert_array_almost_equal(d_marginal_ll, d_marginal_ll_true, d_mll_compare_decimal)
        
    def test_Matern52_kernel(self,):
        """
        This function checks that the functions "marginal_ll" and "test_marginal_ll"
        works identically. This is an example of a good conditioning of the kernel matrix.
        """
        
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_data, y_data = generate_data(n_points,0,1000)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
        
        mll_compare_decimal= 12
        d_mll_compare_decimal = 12
        regularization_type=2
        largest_cond_num=1e13
        
        self.run_with_reg_GP2(x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal, d_mll_compare_decimal, regularization_type, largest_cond_num)
                        
#        self.run_with_reg_GP3(x_data, y_data, kernel1, kernel2, noise_var,
#                        mll_compare_decimal=3, d_mll_compare_decimal=2)

    def test_test_Matern52_kernel(self,):
        """
        This function compares "test_marginal_ll" function and
        standard GPy GP regression. 
        """
        
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_data, y_data = generate_data(n_points,0,1000)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
        
        p_regularization_type = 2
        p_largest_cond_num = 1e13
        
        mll_compare_decimal = 3
        d_mll_compare_decimal = 2
        self.run_with_reg_GP1(x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal, d_mll_compare_decimal, p_regularization_type, p_largest_cond_num)
        
#        self.run_with_reg_GP2(x_data, y_data, kernel1, kernel2, noise_var,
#                mll_compare_decimal, d_mll_compare_decimal)
        
        
    def test_Complex_kernel(self,):
        """
        This function checks that the functions "marginal_ll" and "test_marginal_ll"
        works identically. This is an example of not very good conditioning
        of the kernel matrix.
        """
        
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_data, y_data = generate_data(n_points,0,1000)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale) +\
                    GPy.kern.sde_RBF( 1,variance=variance, lengthscale=lengthscale, approx_order=10, balance=False)
        
        # Very bed conditioning of P_inf example
        #kernel1 = GPy.kern.sde_RBF( 1,variance=variance*5, lengthscale=lengthscale*20, approx_order=10, balance=False)
        
        # kernel1 = GPy.kern.sde_RBF( 1,variance=variance, lengthscale=lengthscale, approx_order=10, balance=False)
        
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale) +\
                    GPy.kern.RBF( 1,variance=variance, lengthscale=lengthscale)
                    
        # kernel2 = GPy.kern.sde_RBF( 1,variance=variance, lengthscale=lengthscale, approx_order=10, balance=False)        
        
        p_regularization_type = 3
        p_largest_cond_num = 1e7
        
        mll_compare_decimal = 10
        d_mll_compare_decimal = 7
        
        self.run_with_reg_GP2(x_data, y_data, kernel1, kernel2, noise_var,
                mll_compare_decimal, d_mll_compare_decimal, p_regularization_type, p_largest_cond_num)
                
#        self.run_with_reg_GP3(x_data, y_data, kernel1, kernel2, noise_var,
#                        mll_compare_decimal=3, d_mll_compare_decimal=1)
                        
    def test_test_Complex_kernel(self,):
        """
        This function compares "test_marginal_ll" function and
        standard GPy GP regression. 
        """
        
        
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_data, y_data = generate_data(n_points,0,1000)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale) +\
                    GPy.kern.sde_RBF( 1,variance=variance, lengthscale=lengthscale, approx_order=10, balance=False)
        
        # Very bed conditioning of P_inf example
        #kernel1 = GPy.kern.sde_RBF( 1,variance=variance*5, lengthscale=lengthscale*20, approx_order=10, balance=False)
        
        # kernel1 = GPy.kern.sde_RBF( 1,variance=variance, lengthscale=lengthscale, approx_order=10, balance=False)
        
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale) +\
                    GPy.kern.RBF( 1,variance=variance, lengthscale=lengthscale)
                    
        # kernel2 = GPy.kern.sde_RBF( 1,variance=variance, lengthscale=lengthscale, approx_order=10, balance=False)        
        
        p_regularization_type = 3
        p_largest_cond_num = 1e7
        
        mll_compare_decimal = 1
        d_mll_compare_decimal = 1
        
        self.run_with_reg_GP1(x_data, y_data, kernel1, kernel2, noise_var,
                mll_compare_decimal, d_mll_compare_decimal, p_regularization_type, p_largest_cond_num)
                        
#        self.run_with_reg_GP3(x_data, y_data, kernel1, kernel2, noise_var,
#                        mll_compare_decimal=3, d_mll_compare_decimal=1)
                        
    def test_mult_Matern_kernel(self,):
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_data, y_data = generate_data(n_points)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)       
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
        
        self.run_with_reg_GP2(x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal=3, d_mll_compare_decimal=1)
    
    
#    def test_Periodic_kernel(self,):
#        np.random.seed(234) # seed the random number generator
#    
#        n_points = 1000
#        x_data, y_data = generate_data(n_points)
#
#        variance = np.random.uniform(0.1, 1.0) # 0.5
#        lengthscale = np.random.uniform(0.2,10) #3.0
#        period = np.random.uniform(0.5,2) #1.0
#        noise_var = np.random.uniform( 0.01, 1) # 0.1
#        
#        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)    
#        kernel2 = GPy.kern.StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale )
#        try:
#            self.run_with_reg_GP(x_data, y_data, kernel1, kernel2, noise_var,
#                        mll_compare_decimal=1, d_mll_compare_decimal=0)
#        except AssertionError:
#            raise SkipTest("Skipping periodic kernel test because regularization is currently removed from the code.")
#   
#    def analyze_differences(arr1, arr2, ignore_relative_small=1e-7 ):
#        """
#
#        Output:
#        ------------------
#        
#        max(abs(arr1-arr2)) absolute differences
#        
#        max(abs(arr1-arr2) / abs(arr1)), relative differences
#        removing small and ignoring zero divisions
#        
#        arr1[argmax]
#        arr2[argmax]
#        """        
#        tmp1 = np.abs( arr1 - arr2)
#        tmp1[ tmp1< np.max(tmp1)*ignore_relative_small ] = 0
        
                
        
    def run_build_matrix(self, X, Y, kernel, noise_var, p_regularization_type, p_largest_cond_num,
                         matrix_compare_decimals, derivative_compare_decimals):
        """
        Test the function build_matices in the btd_inference.
        
        Input:
        ----------------------------
        
        
        """
        #import pdb;pdb.set_trace()
        
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel.sde()
        #import pdb; pdb.set_trace()
        Ki_diag_true, Ki_low_diag_true, Ki_logdet_true, d_Ki_diag_true, d_Ki_low_diag_true, d_Ki_logdet_true, _, _ = \
            btd_inference.test_build_matrices(X, Y, F, L, Qc, P_inf, H, p_largest_cond_num, p_regularization_type, 
                       True, dP_inft, dFt, dQct)
                       
        Ki_diag, Ki_low_diag, Ki_logdet, d_Ki_diag, d_Ki_low_diag, d_Ki_logdet = \
            btd_inference.build_matrices(X, Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, p_regularization_type, 
                       True, dP_inft, dP0t, dFt, dQct)
        
        #import pdb;pdb.set_trace()
        #t1 = np.max(np.abs() )
        t1 = np.max(np.abs(Ki_diag_true- Ki_diag))
        t2 = np.max(np.abs(Ki_low_diag_true- Ki_low_diag) )
        t3 = np.max(np.abs( Ki_logdet_true- Ki_logdet) )
        t4 = np.max(np.abs(d_Ki_diag_true - d_Ki_diag))
        t5 = np.max(np.abs(d_Ki_low_diag_true - d_Ki_low_diag) )
        t6 = np.max(np.abs(d_Ki_logdet_true - d_Ki_logdet))
        
        np.testing.assert_array_almost_equal(Ki_diag_true, Ki_diag, decimal=matrix_compare_decimals)
        np.testing.assert_array_almost_equal(Ki_low_diag_true, Ki_low_diag, decimal=matrix_compare_decimals)
        np.testing.assert_array_almost_equal(Ki_logdet_true, Ki_logdet, decimal=matrix_compare_decimals)
        
        np.testing.assert_array_almost_equal(d_Ki_diag_true, d_Ki_diag, decimal=derivative_compare_decimals)
        np.testing.assert_array_almost_equal(d_Ki_low_diag_true, d_Ki_low_diag, decimal=derivative_compare_decimals)
        np.testing.assert_array_almost_equal(d_Ki_logdet_true, d_Ki_logdet, decimal=derivative_compare_decimals)
        
    def test_build_matrix_Matern52(self,):
        """
        Test the function build_matices in the btd_inference. Matern kernel is used.
        """
        
        np.random.seed(234) # seed the random number generator
    
        n_points = 4
        x_data, y_data = generate_data(n_points,0, 4)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
        
        self.run_build_matrix(x_data, y_data, kernel, noise_var, p_regularization_type=2, p_largest_cond_num=1e12,
                         matrix_compare_decimals=10, derivative_compare_decimals=8)
    
    def test_build_matrix_RBF(self,):
        """
        Test the function build_matices in the btd_inference. Matern kernel is used.
        """
        
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_data, y_data = generate_data(n_points,0, 1000)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel = GPy.kern.sde_RBF( 1,variance=variance, lengthscale=lengthscale, approx_order=10, balance=True)
        
        self.run_build_matrix(x_data, y_data, kernel, noise_var, p_regularization_type=2, p_largest_cond_num=1e12,
                         matrix_compare_decimals=9, derivative_compare_decimals=9)
    
class SparsePrecitionTests(np.testing.TestCase):
    """
    Tests for Scikit-sparse, auxiliaryw functions and mean variance computations
    """
    
    def setUp(self):
        pass

#    def test_tridiag_inv(self):
#        """
#        Block-tridiagonal inversion, find diag. of result.
#        
#        Test solving problem of inverting block-diagonal matrix and  obtaining
#        only the diagonal of the inverse. Mostly test "sparse_inv_rhs" function.
#        """
#        np.random.seed(234) # seed the random number generator
#        
#        n_points = 1000
#        x_data, y_data = generate_data(n_points)
#
#        variance = 0.5
#        lengthscale = 3.0
#        period = 1.0 # For periodic
#        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)  
#        #kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
#        #kernel1 = GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
#        noise_var = 0.1
#        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
#        block_size = F.shape[1]    
#    
#        grad_calc_params = {}
#        grad_calc_params['dP_inf'] = dP_inft
#        grad_calc_params['dF'] = dFt
#        grad_calc_params['dQc'] = dQct
#    
#        (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
#         matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(x_data, 
#                y_data, F, L, Qc, P_inf, P0, H, p_largest_cond_num=1e20, compute_derivatives=True,
#                           grad_calc_params=grad_calc_params, p_regularization_type=2)
#
#        HtH = np.dot(H.T, H)
#        
##        res = sparse_inference.marginal_ll( block_size, y_data, Ait, Qi, \
##        GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, 
##            Kip=Kip, matrix_blocks= matrix_blocks, 
##            matrix_blocks_derivatives = matrix_blocks_derivatives)
#        
#        _, _, tridiag_inv_data = sparse_inference.deriv_determinant( n_points, block_size, HtH, noise_var, 
#                                 matrix_blocks, compute_derivatives=False, deriv_num=None, 
#                                 matrix_derivs=matrix_blocks_derivatives, compute_inv_main_diag=True)
#                                     
#        # Compute the true inversion
#        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
#        Ki = 0.5*(Ki + Ki.T)
#        KiN = Ki +  GtG /noise_var# Precision with a noise        
#        KiN = KiN.toarray()
#        
#        #rhs_block = H.T        
#        #rhs_block = np.eye(block_size)
#        second_block_size = 1        
#        #rhs_block = 100*np.random.rand(block_size*n_points,second_block_size) # block vector
#        #import pdb; pdb.set_trace()
#        rhs_block = np.kron( np.ones( (n_points,1)) , H.T )
#        rhs_matrix = sp.linalg.block_diag(*rhs_block.reshape(n_points, block_size,second_block_size) )   # block diagonal matrix     
#        
#        (block_dim_1, block_dim_2) = (block_size, second_block_size)        
#        
#        #tr_matr = np.linalg.solve(KiN, np.kron( np.eye(n_points), rhs_block) )
#        tr_matr = np.linalg.solve(KiN, rhs_matrix)
#        
#        true_result = np.empty( (block_dim_1*n_points, block_dim_2) )        
#        for i in range(0,n_points):
#            
#            ind1 = i*block_dim_1
#            ind2 = i*block_dim_2
#            true_result[ind1:ind1+block_dim_1,:] =  tr_matr[ind1:ind1+block_dim_1,ind2:ind2+block_dim_2]
#        
#        test_inverse = sparse_inference.sparse_inv_rhs(n_points, block_size, matrix_blocks, HtH/noise_var, None,tridiag_inv_data, rhs_block)
#        
#        max_diff = np.max( np.abs(true_result -  test_inverse))
#        print(max_diff)
#        np.testing.assert_array_almost_equal(max_diff, 0, 7)
#
#    def run_with_reg_GP(self, x_data, y_data, kernel1, kernel2, noise_var, x_test=None,
#                        mean_compare_decimal=2, var_compare_decimal=2):
#        """
#        Main function to compare sparse GP (mean and var) with regular GP predictions.
#        """
#        if x_test is None:
#            x_test = x_data
#            calc_other_var = True
#        else:
#            calc_other_var = False
#        
#        run_sparse = True
#        run_gp = True
#        if run_sparse:
#            
#            sp_mean, sp_var, _ = SparsePrecision1DInference.mean_and_var(kernel1, x_data, y_data, noise_var, x_test, 
#                     p_balance=False, p_largest_cond_num=1e+20, p_regularization_type=2, 
#                     mll_call_tuple=None, mll_call_dict=None, diff_x_crit=None)
#        
#        if run_gp:
#            #print('Regular GP run:')
#            gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
#            gp_mean, gp_var = gp_reg.predict(x_test, include_likelihood=False)            
#        
#        if calc_other_var:
#            n_points = x_data.size
#            #K = gp_reg.posterior.covariance
#            K = gp_reg.kern.K(x_data)
#            #other_mean = np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, y_data ) )
#            #other_var = np.diag( K - np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) ) )[:,None]
#            K_diag = np.diag(K)[:, None]
#            K_result = np.diag(np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) ) )[:, None]
#            other_var = K_diag - K_result
#            #import pdb; pdb.set_trace()
#            
#        #import pdb; pdb.set_trace()
#        print("Mean difference: %e" % (np.max( np.abs( sp_mean - gp_mean )) ))    
#        print("Variance difference: %e"% (np.max( np.abs( sp_var - gp_var )) ))
#        if calc_other_var:
#            print("Manual variance comp. difference: %e"% (np.max( np.abs( other_var - gp_var )) ))        
#        
#        np.testing.assert_array_almost_equal(sp_mean, gp_mean, mean_compare_decimal)
#        np.testing.assert_array_almost_equal(sp_var, gp_var, var_compare_decimal)
#        if calc_other_var:
#            np.testing.assert_array_almost_equal(other_var, gp_var, var_compare_decimal)
#        #plot(x_data, y_data, mean, gp_mean, var, gp_var, np.diag(other_var))
#    
#    def run_with_sde(self, sparse_mode, sde_model, mean_compare_decimal=2, var_compare_decimal=2):
#        """
#        Main function to compare sparse GP (mean and var) with SDE predictions
#        """        
#        
#        sparse_mean, sparse_var = sparse_mode.predict(None)
#        sde_mean, sde_var = sde_model.predict(None)
#        
#        print("Mean difference: %e" % (np.max( np.abs( sparse_mean - sde_mean )) ))    
#        print("Variance difference: %e"% (np.max( np.abs( sparse_var - sde_var )) ))
#
#        np.testing.assert_array_almost_equal(sparse_mean, sde_mean, mean_compare_decimal)
#        np.testing.assert_array_almost_equal(sparse_var, sde_var, var_compare_decimal)
#    
#    def test_predictions_1(self,):
#        """
#        Test SpInGP predictions for the future. (Matern3/2)
#        """
#        
#        n_train_points = 100
#        x_train_data, y_train_data = generate_data(n_train_points)
#        
#        n_test_points = 50
#        x_test_data, y_test_data = generate_data(n_test_points)
#
#        x_test_data = x_test_data + np.max(x_train_data) + 0.1
#        #x_test_data = x_train_data
#
#        variance = np.random.uniform(0.1, 1.0) # 0.5
#        lengthscale = np.random.uniform(0.2,10) #3.0
#        noise_var = np.random.uniform( 0.01, 1) # 0.1
#        
#        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
#        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
#        
#        self.run_with_reg_GP(x_train_data, y_train_data, kernel1, kernel2, noise_var, x_test_data,
#                        mean_compare_decimal=7, var_compare_decimal=8)
#                    
#    def test_predictions_2(self,):
#        """
#        Test SpInGP predictions mixed. (Matern3/2)
#        """
#        
#        n_train_points = 100
#        x_train_data, y_train_data = generate_data(n_train_points)
#        
#        n_test_points = 50
#        x_test_data, y_test_data = generate_data(n_test_points)
#
#        #x_test_data = x_test_data + np.max(x_train_data) + 0.1
#        #x_test_data = x_train_data
#
#        variance = np.random.uniform(0.1, 1.0) # 0.5
#        lengthscale = np.random.uniform(0.2,10) #3.0
#        noise_var = np.random.uniform( 0.01, 1) # 0.1
#        
#        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
#        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
#        
#        self.run_with_reg_GP(x_train_data, y_train_data, kernel1, kernel2, noise_var, x_test_data,
#                        mean_compare_decimal=7, var_compare_decimal=8)
#                        
#    def test_Matern32_kernel(self,):
#        """
#        Compare SpInGP and GP for training points. (Matern3/2)
#        
#        """
#        np.random.seed(234) # seed the random number generator
#    
#        n_points = 100
#        x_data, y_data = generate_data(n_points)
#
#        variance = np.random.uniform(0.1, 1.0) # 0.5
#        lengthscale = np.random.uniform(0.2,10) #3.0
#        noise_var = np.random.uniform( 0.01, 1) # 0.1
#        
#        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
#        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
#        
#        self.run_with_reg_GP(x_data, y_data, kernel1, kernel2, noise_var,
#                        mean_compare_decimal=6, var_compare_decimal=8)
#    
#    def test_sde_sparse_1(self,):    
#        """
#        Matern3/2 prediction compariosn with SDE.        
#        
#        """
#    
#        n_points = 300
#        x_data, y_data = generate_data(n_points)
#
#        variance = np.random.uniform(0.1, 1.0) # 0.5
#        lengthscale = np.random.uniform(0.2,10) #3.0
#        noise_var = 0.1 #np.random.uniform( 0.01, 1) # 0.1
#        
#        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)
#        kernel2 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
#                    
#        #GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
#        sparse_model = ss_sparse_model.SparcePrecisionGP(x_data, y_data, kernel1, noise_var=noise_var )
#        ssm = ss_model.StateSpace(x_data,y_data,kernel2, noise_var=noise_var )
#    
#        self.run_with_sde( sparse_model, ssm, mean_compare_decimal=8, var_compare_decimal=8)
#
#
#
#    def test_sparse_determinant_computation(self,):
#        """    
#        BTD determ. and its derivs. by 2-methods
#        """
#      
#        np.random.seed(234) # seed the random number generator
#            
#        n_points = 2000
#        x_data, y_data = generate_data(n_points)
#    
#        variance = 0.5
#        lengthscale = 3.0
#        period = 1.0 # For periodic
#        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)  
#        #kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
#        #kernel1 = GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
#        noise_var = 0.1
#        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
#        block_size = F.shape[1]    
#    
#        grad_calc_params = {}
#        grad_calc_params['dP_inf'] = dP_inft
#        grad_calc_params['dF'] = dFt
#        grad_calc_params['dQc'] = dQct
#    
#        (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
#         matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(x_data, 
#                y_data, F, L, Qc, P_inf, P0, H, p_largest_cond_num=1e20, compute_derivatives=True,
#                           grad_calc_params=grad_calc_params, p_regularization_type=2)
#                               
#        HtH = np.dot(H.T, H)
#        deriv_num = len(Ki_derivatives) + 1    
#        
#        t1 = time.time()
#        d_log_det_1, log_det_1, tridiag_inv_data = sparse_inference.deriv_determinant( n_points, block_size, HtH, noise_var, 
#                                 matrix_blocks, compute_derivatives=True, deriv_num=deriv_num, 
#                                 matrix_derivs=matrix_blocks_derivatives, compute_inv_main_diag=True)
#        t1 = time.time() - t1 # first  determinant derivative calculation
#    #    rhs = np.zeros((6,2)); 
#    #    rhs[0:2,0:2] = matrix_blocks_derivatives['dAkd'][0][0,:]
#    #    rhs[2:4,0:2] = matrix_blocks_derivatives['dAkd'][1][0,:]
#    #    rhs[4:6,0:2] = matrix_blocks_derivatives['dAkd'][2][0,:]
#    #    
#    #    inv_1 = ss.sparse_inv_rhs(n_points, block_size, matrix_blocks, 
#    #                       HtH/noise_var, None, 
#    #                       tridiag_inv_data, rhs)
#        # Compute the true inversion
#        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
#        Ki = 0.5*(Ki + Ki.T)
#        KiN = Ki +  GtG /noise_var# Precision with a noise        
#        
#        analyzed_factor = cholmod.analyze(KiN) # perform only once this operation
#                                                  # since this is expensive?  
#        
#        #import pdb; pdb.set_trace()
#        KiN_factor = analyzed_factor._clone()
#        KiN_factor.cholesky_inplace(KiN)
#        log_det_2 = KiN_factor.logdet()
#        
#        ii1 = Ki_derivatives[0]
#        #ii1[0:2, 2:4] = np.zeros((2,2));  ii1[2:4, 4:6] = np.zeros((2,2)); 
#        #ii1[2:4, 0:2] = np.zeros((2,2));  ii1[4:6, 2:4] = np.zeros((2,2))    
#        
#        
#        inv2 = KiN_factor.solve_A(ii1)
#        inv2 = inv2.todense()
#        
#        t2 = time.time()
#        d_log_det_3, log_det_3, inv_diag_3 = sparse_inference.deriv_determinant2( n_points, block_size, HtH, noise_var, 
#                                     matrix_blocks, matrix_blocks_derivatives, front_multiplier=None,
#                                     compute_inv_diag=False, add_noise_deriv=True)    
#        t2 = time.time() - t2 # first  determinant derivative calculation
#        
#        print("Determ. deriv. calc. (%i samples) t1(method 1) = %f , t2(method 2) = %f" % (n_points, t1, t2))
#        print("Sparse routive logdet - Thomas logdet =", ( np.round( np.abs( log_det_2 - log_det_3),4),) )
#        
#        print("L1( d_loddet1 - d_loddet2) ="); print( np.abs( d_log_det_1 - d_log_det_3) )
#        np.testing.assert_array_almost_equal(np.abs( d_log_det_1 - d_log_det_3) , 0, 9)
#        #import pdb; pdb.set_trace()
#        #pass
    
    def run_mean_var_prepare_matrices(self, x_train_data, y_train_data, x_test_data, 
                                  kernel, noise_var, p_largest_cond_num, 
                                  p_regularization_type,compare_decimal):
        """

        """
        
        #import pdb; pdb.set_trace()
        if x_test_data is None:
            X = x_train_data
            Y = y_train_data
            which_train_true = np.ones(x_train_data.shape)
        else:
            X = np.vstack( (x_train_data, x_test_data) )
            Y = np.vstack( (y_train_data, np.zeros( (x_test_data.shape[0],1) ) ) )
            which_train_true = np.vstack( (np.ones(x_train_data.shape), np.zeros(x_test_data.shape) )    )
        
        _, forw_index, back_index = np.unique( X, True, True )
        X = X[forw_index]
        Y = Y[forw_index]
        which_train_true = which_train_true[forw_index]
        
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel.sde()
        block_size = F.shape[1]   
        
        # Kernel <-
                                       
        Ki_diag_true, Ki_low_diag_true, _,_,_,_ = btd_inference.build_matrices(X, Y, F, L, Qc, P_inf, P0, H, p_largest_cond_num, p_regularization_type, 
                       compute_derivatives=False, dP_inf=None, dP0=None, dF=None, dQc=None)
                       
                       
        Ki_diag, Ki_low_diag, _,_,_ = btd_inference.mean_var_calc_prepare_matrices(block_size, x_train_data, x_test_data, 
                                     y_train_data, noise_var, F, L, Qc, P_inf, P0, H,
                                       p_largest_cond_num, p_regularization_type, diff_x_crit=None)        
                                       
        #import pdb; pdb.set_trace()
        pass
        np.testing.assert_array_almost_equal(Ki_diag_true, Ki_diag, compare_decimal)
        np.testing.assert_array_almost_equal(Ki_low_diag_true, Ki_low_diag, compare_decimal)
        #np.testing.assert_array_almost_equal(which_train_true, which_train, compare_decimal)
        
    def test_mean_var_prepare_matrices1(self,):
        """
        Tests the function "mean_var_prepare_matrices" from btd_inference.
        
        General random generation case
        """
    
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_train_data, y_train_data = generate_data(n_points,0,1000)
        
        x_test_data, _ = generate_data(int(n_points/5),0,1000)
        
        
        # Kernel ->
        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)   
        
        p_largest_cond_num = 1e13
        p_regularization_type = 2
        compare_decimal = 12
        
        self.run_mean_var_prepare_matrices(x_train_data, y_train_data, x_test_data, 
                                  kernel, noise_var, p_largest_cond_num, 
                                  p_regularization_type,compare_decimal)
    
    def test_mean_var_prepare_matrices2(self,):
        """
        Tests the function "mean_var_prepare_matrices" from btd_inference.
        
        No test inputs, only training.
        """
    
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_train_data, y_train_data = generate_data(n_points,0,1000)
        
        x_test_data = None
        
        # Kernel ->
        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)   
        
        p_largest_cond_num = 1e13
        p_regularization_type = 2
        compare_decimal = 15
        
        self.run_mean_var_prepare_matrices(x_train_data, y_train_data, x_test_data, 
                                  kernel, noise_var, p_largest_cond_num, 
                                  p_regularization_type,compare_decimal)
                                  
    def test_mean_var_prepare_matrices3(self,):
        """
        Tests the function "mean_var_prepare_matrices" from btd_inference.
        
        Some intersecting points between training and test sets.
        """
    
        np.random.seed(234) # seed the random number generator
    
        
        x_train_data = np.array( ((1,2,3,4.0),) ).T
        y_train_data = np.array( ((1.1,2.2,3.3,4.4),) ).T
        
        x_test_data = np.array( ((1,1.5,2),) ).T
        
        # Kernel ->
        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)   
        
        p_largest_cond_num = 1e13
        p_regularization_type = 2
        compare_decimal = 12
        
        self.run_mean_var_prepare_matrices(x_train_data, y_train_data, x_test_data, 
                                  kernel, noise_var, p_largest_cond_num, 
                                  p_regularization_type,compare_decimal)
                                  
                                  
    def run_mean_var_computation(self,x_train_data, y_train_data, x_test_data, 
                                  noise_var, kernel_sparse, kernel_gp,
                                  p_largest_cond_num, p_regularization_type,
                                  mean_compare_decimal, var_compare_decimal):
        """

        """
        #import pdb; pdb.set_trace()
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel_sparse.sde()
        block_size = F.shape[1]   
        
        Ki_diag, Ki_low_diag, test_points_num, forward_index, inverse_index = \
                btd_inference.mean_var_calc_prepare_matrices(block_size, x_train_data, x_test_data, 
                                     y_train_data, noise_var, F, L, Qc, P_inf, P0, H,
                                       p_largest_cond_num, p_regularization_type, diff_x_crit=None)
                                       
        sp_mean, sp_var = btd_inference.mean_var_calc(block_size, y_train_data, Ki_diag, 
                                                      Ki_low_diag, H, noise_var, test_points_num, forward_index, inverse_index)
    
        #import pdb; pdb.set_trace()
        #print('Regular GP run:')
        gp_reg = GPy.models.GPRegression(x_train_data, y_train_data, kernel_gp, noise_var=noise_var)
        gp_mean_true, gp_var_true = gp_reg.predict(x_test_data, include_likelihood=False)            
        
        #import pdb; pdb.set_trace()
        print("Mean difference: %e" % (np.max( np.abs( sp_mean - gp_mean_true )) ))    
        print("Variance difference: %e"% (np.max( np.abs( sp_var - gp_var_true )) ))       
        
        np.testing.assert_array_almost_equal(sp_mean, gp_mean_true, mean_compare_decimal)
        np.testing.assert_array_almost_equal(sp_var, gp_var_true, var_compare_decimal)
       
    def test_mean_var_calc(self, ):
        """

        """

        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_train_data, y_train_data = generate_data(n_points,0,1000)
        
        x_test_data, _ = generate_data(int(n_points/5),0,1000)
        #x_test_data, _ = generate_data(5,0,1000)
        
        # Kernel ->
        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)        
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale) 
        
        p_largest_cond_num = 1e13
        p_regularization_type = 2
        mean_compare_decimal = 5
        var_compare_decimal = 5
        
        self.run_mean_var_computation(x_train_data, y_train_data, x_test_data, 
                                  noise_var, kernel1, kernel2,
                                  p_largest_cond_num, p_regularization_type,
                                  mean_compare_decimal, var_compare_decimal)
                                  
if __name__ == "__main__":
    print("Running sparse precision inference tests...")
    unittest.main()
    
    
    
    
#    tt = ComputationalRoutinesTests('test_btd_system_solution1')
#    tt.test_btd_system_solution1()
#    
#    tt = ComputationalRoutinesTests('test_btd_system_solution2')
#    tt.test_btd_system_solution2()
#    
#    tt = ComputationalRoutinesTests('test_reg_system_solution')
#    tt.test_reg_system_solution()
#    
#    tt = ComputationalRoutinesTests('test_TransposeSubmatrices')
#    tt.test_TransposeSubmatrices()
#    
#    tt = ComputationalRoutinesTests('test_BTD_multiplication')
#    tt.test_BTD_multiplication()
#
#    tt = SparsePrecitionMLLTests('test_test_Matern52_kernel')
#    tt.test_test_Matern52_kernel()
#
#    tt = SparsePrecitionMLLTests('test_test_Complex_kernel')
#    tt.test_test_Complex_kernel()
#
#    tt = SparsePrecitionMLLTests('test_build_matrix_Matern52')
#    tt.test_build_matrix_Matern52()
# 
#    tt = SparsePrecitionMLLTests('test_build_matrix_RBF')
#    tt.test_build_matrix_RBF()
#    
#    tt = SparsePrecitionMLLTests('test_Matern52_kernel')
#    tt.test_Matern52_kernel()
# 
#    tt = SparsePrecitionMLLTests('test_Complex_kernel')
#    tt.test_Complex_kernel()
#     
#    tt = SparsePrecitionTests('test_mean_var_prepare_matrices1')
#    tt.test_mean_var_prepare_matrices1()
#    
#    tt = SparsePrecitionTests('test_mean_var_prepare_matrices2')
#    tt.test_mean_var_prepare_matrices2()
#    
#    tt = SparsePrecitionTests('test_mean_var_prepare_matrices3')
#    tt.test_mean_var_prepare_matrices3()
#    
#    tt = SparsePrecitionTests('test_mean_var_calc')
#    tt.test_mean_var_calc()
    
    
    
    
    
    #tt = SparsePrecitionTests('test_sde_sparse_1')
    #tt.test_tridiag_inv()
    #tt.test_sparse_determinant_computation()
    #tt.test_predictions_1()
    #tt.test_Matern32_kernel()
    #tt.test_sde_sparse_1()
    
    #tt = SparsePrecitionMLLTests('test_mult_Matern_kernel')
    #tt.test_Matern32_kernel()
    #tt.test_Matern52_kernel()
    #tt.test_mult_Matern_kernel()
    #tt.test_Periodic_kernel()
    
        
        
        
        