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
#import GPy.models.state_space_model as SS_model
#from .state_space_main_tests import generate_x_points, generate_sine_data, \
#    generate_linear_data, generate_brownian_data, generate_linear_plus_sin

import time
import GPy.models.state_space_model as ss_model
import GPy.models.ss_sparse_model as ss_sparse_model

from nose import SkipTest

from GPy.inference.latent_function_inference.ss_sparse_inference import sparse_inference
from GPy.inference.latent_function_inference.ss_sparse_inference import SparsePrecision1DInference

def generate_data(n_points):
    """
    Input:
    -----------
    
    n_points: number of data points.
    """

    x_lower_value = 0.0
    x_upper_value = 200.0

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

class SparsePrecitionMLLTests(np.testing.TestCase):
    """
    The file tests marginal likelihood (MLL) and its derivatives calculation.    
    
    """
    
    def setUp(self):
        pass

    def run_with_reg_GP(self, x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal=2, d_mll_compare_decimal=2):
        """
        The main function which is run for different settings.
        """

        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
        block_size = F.shape[1]    
        
        grad_calc_params = {}
        grad_calc_params['dP_inf'] = dP_inft
        grad_calc_params['dF'] = dFt
        grad_calc_params['dQc'] = dQct
        
        # Do not change with repetiotions
        
        mll_diff = None
        d_mll_diff = None
        #d_mll_diff_relative = None
        
        run_sparse = True
        run_gp = True
        if run_sparse:
            #print('Sparse GP run:')
            (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
             matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(x_data, 
                    y_data, F, L, Qc, P_inf, P0, H, p_largest_cond_num=1e20, compute_derivatives=True,
                               grad_calc_params=grad_calc_params, p_regularization_type=2)
            
                               
            res = sparse_inference.marginal_ll( block_size, y_data, Ait, Qi, \
            GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, 
                Kip=Kip, matrix_blocks= matrix_blocks, 
                matrix_blocks_derivatives = matrix_blocks_derivatives)
            
            marginal_ll = res[0]; d_marginal_ll = res[1]

        if run_gp:
            #print('Regular GP run:')
            gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
            
        mll_diff = marginal_ll - (-gp_reg.objective_function())
        reg_d_mll = -gp_reg.objective_function_gradients(); reg_d_mll.shape = (reg_d_mll.shape[0],1)
        d_mll_diff = d_marginal_ll - (reg_d_mll)
        #d_mll_diff_relative = np.sum( np.abs(d_mll_diff) ) / np.sum( np.abs(reg_d_mll) ) 

        np.testing.assert_array_almost_equal(marginal_ll, -gp_reg.objective_function(), mll_compare_decimal)
        np.testing.assert_array_almost_equal(d_marginal_ll, reg_d_mll, d_mll_compare_decimal)

    def test_Matern32_kernel(self,):
        
        np.random.seed(234) # seed the random number generator
    
        n_points = 1000
        x_data, y_data = generate_data(n_points)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
        
        self.run_with_reg_GP(x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal=3, d_mll_compare_decimal=2)

    def test_Matern52_kernel(self,):
        np.random.seed(234) # seed the random number generator
    
        n_points = 1000
        x_data, y_data = generate_data(n_points)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
        
        self.run_with_reg_GP(x_data, y_data, kernel1, kernel2, noise_var,
                        mll_compare_decimal=3, d_mll_compare_decimal=1)

    def test_mult_Matern_kernel(self,):
        np.random.seed(234) # seed the random number generator
    
        n_points = 1000
        x_data, y_data = generate_data(n_points)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)       
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
        
        self.run_with_reg_GP(x_data, y_data, kernel1, kernel2, noise_var,
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
    
class SparsePrecitionTests(np.testing.TestCase):
    """
    Tests for Scikit-sparse, auxiliaryw functions and mean variance computations
    """
    
    def setUp(self):
        pass

    def test_tridiag_inv(self):
        """
        Block-tridiagonal inversion, find diag. of result.
        
        Test solving problem of inverting block-diagonal matrix and  obtaining
        only the diagonal of the inverse. Mostly test "sparse_inv_rhs" function.
        """
        np.random.seed(234) # seed the random number generator
        
        n_points = 1000
        x_data, y_data = generate_data(n_points)

        variance = 0.5
        lengthscale = 3.0
        period = 1.0 # For periodic
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)  
        #kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
        #kernel1 = GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
        noise_var = 0.1
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
        block_size = F.shape[1]    
    
        grad_calc_params = {}
        grad_calc_params['dP_inf'] = dP_inft
        grad_calc_params['dF'] = dFt
        grad_calc_params['dQc'] = dQct
    
        (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
         matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(x_data, 
                y_data, F, L, Qc, P_inf, P0, H, p_largest_cond_num=1e20, compute_derivatives=True,
                           grad_calc_params=grad_calc_params, p_regularization_type=2)

        HtH = np.dot(H.T, H)
        
#        res = sparse_inference.marginal_ll( block_size, y_data, Ait, Qi, \
#        GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, 
#            Kip=Kip, matrix_blocks= matrix_blocks, 
#            matrix_blocks_derivatives = matrix_blocks_derivatives)
        
        _, _, tridiag_inv_data = sparse_inference.deriv_determinant( n_points, block_size, HtH, noise_var, 
                                 matrix_blocks, compute_derivatives=False, deriv_num=None, 
                                 matrix_derivs=matrix_blocks_derivatives, compute_inv_main_diag=True)
                                     
        # Compute the true inversion
        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
        Ki = 0.5*(Ki + Ki.T)
        KiN = Ki +  GtG /noise_var# Precision with a noise        
        KiN = KiN.toarray()
        
        #rhs_block = H.T        
        #rhs_block = np.eye(block_size)
        second_block_size = 1        
        #rhs_block = 100*np.random.rand(block_size*n_points,second_block_size) # block vector
        #import pdb; pdb.set_trace()
        rhs_block = np.kron( np.ones( (n_points,1)) , H.T )
        rhs_matrix = sp.linalg.block_diag(*rhs_block.reshape(n_points, block_size,second_block_size) )   # block diagonal matrix     
        
        (block_dim_1, block_dim_2) = (block_size, second_block_size)        
        
        #tr_matr = np.linalg.solve(KiN, np.kron( np.eye(n_points), rhs_block) )
        tr_matr = np.linalg.solve(KiN, rhs_matrix)
        
        true_result = np.empty( (block_dim_1*n_points, block_dim_2) )        
        for i in range(0,n_points):
            
            ind1 = i*block_dim_1
            ind2 = i*block_dim_2
            true_result[ind1:ind1+block_dim_1,:] =  tr_matr[ind1:ind1+block_dim_1,ind2:ind2+block_dim_2]
        
        test_inverse = sparse_inference.sparse_inv_rhs(n_points, block_size, matrix_blocks, HtH/noise_var, None,tridiag_inv_data, rhs_block)
        
        max_diff = np.max( np.abs(true_result -  test_inverse))
        print(max_diff)
        np.testing.assert_array_almost_equal(max_diff, 0, 7)

    def run_with_reg_GP(self, x_data, y_data, kernel1, kernel2, noise_var, x_test=None,
                        mean_compare_decimal=2, var_compare_decimal=2):
        """
        Main function to compare sparse GP (mean and var) with regular GP predictions.
        """
        if x_test is None:
            x_test = x_data
            calc_other_var = True
        else:
            calc_other_var = False
        
        run_sparse = True
        run_gp = True
        if run_sparse:
            
            sp_mean, sp_var, _ = SparsePrecision1DInference.mean_and_var(kernel1, x_data, y_data, noise_var, x_test, 
                     p_balance=False, p_largest_cond_num=1e+20, p_regularization_type=2, 
                     mll_call_tuple=None, mll_call_dict=None, diff_x_crit=None)
        
        if run_gp:
            #print('Regular GP run:')
            gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
            gp_mean, gp_var = gp_reg.predict(x_test, include_likelihood=False)            
        
        if calc_other_var:
            n_points = x_data.size
            #K = gp_reg.posterior.covariance
            K = gp_reg.kern.K(x_data)
            #other_mean = np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, y_data ) )
            #other_var = np.diag( K - np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) ) )[:,None]
            K_diag = np.diag(K)[:, None]
            K_result = np.diag(np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) ) )[:, None]
            other_var = K_diag - K_result
            #import pdb; pdb.set_trace()
            
        #import pdb; pdb.set_trace()
        print("Mean difference: %e" % (np.max( np.abs( sp_mean - gp_mean )) ))    
        print("Variance difference: %e"% (np.max( np.abs( sp_var - gp_var )) ))
        if calc_other_var:
            print("Manual variance comp. difference: %e"% (np.max( np.abs( other_var - gp_var )) ))        
        
        np.testing.assert_array_almost_equal(sp_mean, gp_mean, mean_compare_decimal)
        np.testing.assert_array_almost_equal(sp_var, gp_var, var_compare_decimal)
        if calc_other_var:
            np.testing.assert_array_almost_equal(other_var, gp_var, var_compare_decimal)
        #plot(x_data, y_data, mean, gp_mean, var, gp_var, np.diag(other_var))
    
    def run_with_sde(self, sparse_mode, sde_model, mean_compare_decimal=2, var_compare_decimal=2):
        """
        Main function to compare sparse GP (mean and var) with SDE predictions
        """        
        
        sparse_mean, sparse_var = sparse_mode.predict(None)
        sde_mean, sde_var = sde_model.predict(None)
        
        print("Mean difference: %e" % (np.max( np.abs( sparse_mean - sde_mean )) ))    
        print("Variance difference: %e"% (np.max( np.abs( sparse_var - sde_var )) ))

        np.testing.assert_array_almost_equal(sparse_mean, sde_mean, mean_compare_decimal)
        np.testing.assert_array_almost_equal(sparse_var, sde_var, var_compare_decimal)
    
    def test_predictions_1(self,):
        """
        Test SpInGP predictions for the future. (Matern3/2)
        """
        
        n_train_points = 100
        x_train_data, y_train_data = generate_data(n_train_points)
        
        n_test_points = 50
        x_test_data, y_test_data = generate_data(n_test_points)

        x_test_data = x_test_data + np.max(x_train_data) + 0.1
        #x_test_data = x_train_data

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
        
        self.run_with_reg_GP(x_train_data, y_train_data, kernel1, kernel2, noise_var, x_test_data,
                        mean_compare_decimal=7, var_compare_decimal=8)
                    
    def test_predictions_2(self,):
        """
        Test SpInGP predictions mixed. (Matern3/2)
        """
        
        n_train_points = 100
        x_train_data, y_train_data = generate_data(n_train_points)
        
        n_test_points = 50
        x_test_data, y_test_data = generate_data(n_test_points)

        #x_test_data = x_test_data + np.max(x_train_data) + 0.1
        #x_test_data = x_train_data

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
        
        self.run_with_reg_GP(x_train_data, y_train_data, kernel1, kernel2, noise_var, x_test_data,
                        mean_compare_decimal=7, var_compare_decimal=8)
                        
    def test_Matern32_kernel(self,):
        """
        Compare SpInGP and GP for training points. (Matern3/2)
        
        """
        np.random.seed(234) # seed the random number generator
    
        n_points = 100
        x_data, y_data = generate_data(n_points)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
        
        self.run_with_reg_GP(x_data, y_data, kernel1, kernel2, noise_var,
                        mean_compare_decimal=6, var_compare_decimal=8)
    
    def test_sde_sparse_1(self,):    
        """
        Matern3/2 prediction compariosn with SDE.        
        
        """
    
        n_points = 300
        x_data, y_data = generate_data(n_points)

        variance = np.random.uniform(0.1, 1.0) # 0.5
        lengthscale = np.random.uniform(0.2,10) #3.0
        noise_var = 0.1 #np.random.uniform( 0.01, 1) # 0.1
        
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)
        kernel2 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
                    
        #GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
        sparse_model = ss_sparse_model.SparcePrecisionGP(x_data, y_data, kernel1, noise_var=noise_var )
        ssm = ss_model.StateSpace(x_data,y_data,kernel2, noise_var=noise_var )
    
        self.run_with_sde( sparse_model, ssm, mean_compare_decimal=8, var_compare_decimal=8)



    def test_sparse_determinant_computation(self,):
        """    
        BTD determ. and its derivs. by 2-methods
        """
      
        np.random.seed(234) # seed the random number generator
            
        n_points = 2000
        x_data, y_data = generate_data(n_points)
    
        variance = 0.5
        lengthscale = 3.0
        period = 1.0 # For periodic
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)  
        #kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
        #kernel1 = GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
        noise_var = 0.1
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
        block_size = F.shape[1]    
    
        grad_calc_params = {}
        grad_calc_params['dP_inf'] = dP_inft
        grad_calc_params['dF'] = dFt
        grad_calc_params['dQc'] = dQct
    
        (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
         matrix_blocks_derivatives) = sparse_inference.sparse_inverse_cov(x_data, 
                y_data, F, L, Qc, P_inf, P0, H, p_largest_cond_num=1e20, compute_derivatives=True,
                           grad_calc_params=grad_calc_params, p_regularization_type=2)
                               
        HtH = np.dot(H.T, H)
        deriv_num = len(Ki_derivatives) + 1    
        
        t1 = time.time()
        d_log_det_1, log_det_1, tridiag_inv_data = sparse_inference.deriv_determinant( n_points, block_size, HtH, noise_var, 
                                 matrix_blocks, compute_derivatives=True, deriv_num=deriv_num, 
                                 matrix_derivs=matrix_blocks_derivatives, compute_inv_main_diag=True)
        t1 = time.time() - t1 # first  determinant derivative calculation
    #    rhs = np.zeros((6,2)); 
    #    rhs[0:2,0:2] = matrix_blocks_derivatives['dAkd'][0][0,:]
    #    rhs[2:4,0:2] = matrix_blocks_derivatives['dAkd'][1][0,:]
    #    rhs[4:6,0:2] = matrix_blocks_derivatives['dAkd'][2][0,:]
    #    
    #    inv_1 = ss.sparse_inv_rhs(n_points, block_size, matrix_blocks, 
    #                       HtH/noise_var, None, 
    #                       tridiag_inv_data, rhs)
        # Compute the true inversion
        Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
        Ki = 0.5*(Ki + Ki.T)
        KiN = Ki +  GtG /noise_var# Precision with a noise        
        
        analyzed_factor = cholmod.analyze(KiN) # perform only once this operation
                                                  # since this is expensive?  
        
        #import pdb; pdb.set_trace()
        KiN_factor = analyzed_factor._clone()
        KiN_factor.cholesky_inplace(KiN)
        log_det_2 = KiN_factor.logdet()
        
        ii1 = Ki_derivatives[0]
        #ii1[0:2, 2:4] = np.zeros((2,2));  ii1[2:4, 4:6] = np.zeros((2,2)); 
        #ii1[2:4, 0:2] = np.zeros((2,2));  ii1[4:6, 2:4] = np.zeros((2,2))    
        
        
        inv2 = KiN_factor.solve_A(ii1)
        inv2 = inv2.todense()
        
        t2 = time.time()
        d_log_det_3, log_det_3, inv_diag_3 = sparse_inference.deriv_determinant2( n_points, block_size, HtH, noise_var, 
                                     matrix_blocks, matrix_blocks_derivatives, front_multiplier=None,
                                     compute_inv_diag=False, add_noise_deriv=True)    
        t2 = time.time() - t2 # first  determinant derivative calculation
        
        print("Determ. deriv. calc. (%i samples) t1(method 1) = %f , t2(method 2) = %f" % (n_points, t1, t2))
        print("Sparse routive logdet - Thomas logdet =", ( np.round( np.abs( log_det_2 - log_det_3),4),) )
        
        print("L1( d_loddet1 - d_loddet2) ="); print( np.abs( d_log_det_1 - d_log_det_3) )
        np.testing.assert_array_almost_equal(np.abs( d_log_det_1 - d_log_det_3) , 0, 9)
        #import pdb; pdb.set_trace()
        #pass
        
if __name__ == "__main__":
    print("Running sparse precision inference tests...")
    unittest.main()
    
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
    
        
        
        
        