# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:49:10 2016

@author: alex
"""
import numpy as np
import time
from collections import Iterable
import scipy as sp 
import sksparse.cholmod as cholmod
#import GPy.inference.latent_function_inference.ss_sparse_inference as ss
#import ss_sparse_inference as ss
from GPy.inference.latent_function_inference import ss_sparse_inference
ss = ss_sparse_inference.sparse_inference

import GPy

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
 
def test_random_matr_slicing_speeds(block_size, block_num, rep_number):
    """


    """    
    import time
    constr_dic = {}
    reading_dic = {}
    
    # Constructing a matrix:
    Block = np.random.random( (block_size,block_size) )     
    A_size = block_size * block_num
    # lil
    
    formats = ('lil', 'dok')
    
    for ff in formats:
        init_matr = sp.sparse.random(A_size, A_size)
        
        matr = init_matr.asformat(ff)
        
        constr_dic[ff] = []
        
        for rr in range(0,rep_number):
                                    
            t1 = time.time()                    
            matr[0:block_size, 0:block_size] = Block
            for ii in range(0,block_num-1):
                low_ind_start = ii*block_size
                low_ind_end = low_ind_start + block_size
                
                high_ind_start = (ii+1)*block_size
                high_ind_end =  high_ind_start + block_size
                
                matr[high_ind_start:high_ind_end, high_ind_start:high_ind_end] = Block + Block.T
                matr[low_ind_start:low_ind_end, high_ind_start:high_ind_end] = Block
                matr[high_ind_start:high_ind_end, low_ind_start:low_ind_end] = Block.T
            constr_dic[ff].append(time.time() - t1)

        constr_dic[ff] = np.mean( constr_dic[ff] )

    # Reading a matrix:
    formats = ('lil', 'csc', 'dok')
    
    m1 = matr.copy()
    
    for ff in formats:
        matr = m1.copy().asformat(ff)
        
        reading_dic[ff] = []
        
        for rr in range(0,rep_number):
                                    
            t2 = time.time()                    
            tmp = matr[0:block_size, 0:block_size]
            for ii in range(0,block_num-1):
                low_ind_start = ii*block_size
                low_ind_end = low_ind_start + block_size
                
                high_ind_start = (ii+1)*block_size
                high_ind_end =  high_ind_start + block_size
                
                tmp1 = matr[high_ind_start:high_ind_end, high_ind_start:high_ind_end]
                tmp2 = matr[low_ind_start:low_ind_end, high_ind_start:high_ind_end]
                tmp3 = matr[high_ind_start:high_ind_end, low_ind_start:low_ind_end]
                
            reading_dic[ff].append(time.time() - t2)

        reading_dic[ff] = np.mean( reading_dic[ff] )
    
    # Test matrix conversions:
    m2 = m1.asformat('lil')
    
    reading_dic['lil_to_csc'] = []
    for rr in range(0,rep_number):
        t3 = time.time()  
        tmp1 = m2.tocsc()   
        reading_dic['lil_to_csc'].append(time.time() - t3)
    reading_dic['lil_to_csc'] = np.mean( reading_dic['lil_to_csc'] )
    
    
    m2 = m1.asformat('csc')
    
    reading_dic['csc_to_lil'] = []
    for rr in range(0,rep_number):
        t3 = time.time()  
        tmp1 = m2.tolil()   
        reading_dic['csc_to_lil'].append(time.time() - t3)
    reading_dic['csc_to_lil'] = np.mean( reading_dic['csc_to_lil'] )


    return constr_dic,reading_dic


    
    
    
def test_matrix_cook_book_f164():
    """

    """

    n_points = 100
    x_data, y_data = generate_data(n_points)

    variance = np.random.uniform(0.1, 1.0) # 0.5
    lengthscale = np.random.uniform(0.2,10) #3.0
    noise_var = 0.1 #np.random.uniform( 0.01, 1) # 0.1
    
    kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
    
    gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
    gp_mean, gp_var = gp_reg.predict(x_data, include_likelihood=False)            

    #K = gp_reg.posterior.covariance
    K = gp_reg.kern.K(x_data)
    #other_mean = np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, y_data ) )
    #other_var = np.diag( K - np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) ) )[:,None]
    K_diag = K
    K_result = np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) )
    var1 = K_diag - K_result
    
    var2 = np.eye(n_points)*noise_var - noise_var*np.linalg.solve(K + np.eye(n_points)*noise_var, np.eye(n_points)*noise_var)

    #import pdb; pdb.set_trace()
    print( np.max(np.abs(var1 - var2)) )


def test_sparse_inverse_cov_simple():
    """
    """
    
    # Matern32 ->
#    variance = float(1)
#    lengthscale = float(1)
#    
#    foo  = np.sqrt(3.)/lengthscale 
#    F    = np.array(((0, 1.0), (-foo**2, -2*foo))) 
#    L    = np.array(( (0,), (1.0,) ))
#    Qc   = np.array(((12.*np.sqrt(3) / lengthscale**3 * variance,),)) 
#    H    = np.array(((1.0, 0),)) 
#    Pinf = np.array(((variance, 0.0), (0.0, 3.*variance/(lengthscale**2))))
#    P0 = Pinf.copy()
    # Matern32 <-
    measure_timings = True    
    
    kernel1 = GPy.kern.sde_Matern32(1,variance=0.5, lengthscale=3.)    
    
    (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
    
    grad_calc_params = {}
    grad_calc_params['dP_inf'] = dP_inft
    grad_calc_params['dF'] = dFt
    grad_calc_params['dQc'] = dQct
        
    X = np.array( ((1.0,1.6, 1.8, 1.9),) ).T
    Y = np.array( ((3.0, 4.0, 5.0, 3.2),) ).T
    #import importlib
    #importlib.reload(ss)
    (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Ki) = ss.sparse_inference.sparse_inverse_cov(X, Y, F, L, Qc, P_inf, P0, H, compute_derivatives=True,
                           grad_calc_params=grad_calc_params)
    
    block_size = F.shape[0]
    noise_var = 0.1
    if measure_timings:
        marginal_ll, d_marginal_ll, meas_times, meas_times_desc, sparsity_meas, \
                sparsity_meas_descr = ss.sparse_inference.marginal_ll(block_size, Y, Ait, Qi, \
            GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, Kip=Ki)
    else:
        marginal_ll, d_marginal_ll = ss.sparse_inference.marginal_ll(block_size, Y, Ait, Qi, \
            GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, Kip=Ki)
    print('Sparse GP:')
    print(marginal_ll)
    print(d_marginal_ll) 
    
    kernel2 = GPy.kern.Matern32(1,variance=0.5, lengthscale=3.)
    gp_reg = GPy.models.GPRegression(X,Y,kernel2, noise_var=noise_var)
    print('Regular GP:')
    print(-gp_reg.objective_function())
    print(-gp_reg.objective_function_gradients())         

run_sparse = True    
run_gp = True
do_mean_var_prediction = True  
def test_sparse_gp_timings(n_points, repetitions_num, kernel_num = 0):
    """
    This function returns necessary time measurements for sparse GP marginal
    likelihood and its derivatives computations

    """
    global run_sparse    
    global run_gp
    global do_mean_var_prediction  
    
    
    print('Sparse GP test %i' % n_points)
    x_data, y_data = generate_data(n_points)
    
    variance = 0.5
    lengthscale = 3.0
    period = 1.0 # For periodic
    if (kernel_num == 0): # blocksize is 2
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
    elif (kernel_num == 1): # blocksize is 3
        kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
    elif (kernel_num == 2): # blocksize is 6
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)       
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
    elif (kernel_num == 3): # blocksize is 10
        kernel1 = GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
        kernel2 = GPy.kern.RBF(1,variance=variance, lengthscale=lengthscale)        
    elif (kernel_num == 4): # blocksize is 14 
        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)    
        kernel2 = GPy.kern.StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale )
    elif (kernel_num == 5): # blocksize is 20
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale) * \
            GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale) * \
            GPy.kern.RBF(1,variance=variance, lengthscale=lengthscale) 
    elif (kernel_num == 6): # blocksize is 28
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale) * \
                GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)
        kernel2 =  GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale) * \
                    GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)
    elif (kernel_num == 7): # blocksize is 42
        kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale) * \
                GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)  
        kernel2 = GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale) * \
                GPy.kern.StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale )
    
    
    noise_var = 0.1
    (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
    block_size = F.shape[1]    
    
    grad_calc_params = {}
    grad_calc_params['dP_inf'] = dP_inft
    grad_calc_params['dF'] = dFt
    grad_calc_params['dQc'] = dQct
    
    timings_list = []
    total_timings = {};total_timings[0] = []; total_timings[1] = []; total_timings[2] = [];
    measurements = {}
    if do_mean_var_prediction:
        total_timings[3] = []; total_timings[4] = [];
    # Do not change with repetiotions
    timings_descr = None
    sparsity_info = None
    sparsity_descr = None
    
    for rr in range(0, repetitions_num):
        if run_sparse:
            print('Sparse GP run:')
            t1 = time.time()
            (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
             matrix_blocks_derivatives) = ss.sparse_inverse_cov(x_data, 
                    y_data, F, L, Qc, P_inf, P0, H, compute_derivatives=True,
                               grad_calc_params=grad_calc_params)
            total_timings[0].append(time.time() - t1)
            
            t1 = time.time()
            res = ss.marginal_ll( block_size, y_data, Ait, Qi, \
            GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, 
                Kip=Kip, matrix_blocks= matrix_blocks, 
                matrix_blocks_derivatives = matrix_blocks_derivatives)
            total_timings[1].append(time.time() - t1)        
            marginal_ll = res[0]; d_marginal_ll = res[1]
            
            timings_list.append( res[2])
            timings_descr = res[3]
            sparsity_info = res[4]
            sparsity_descr = res[5]
            
            #import pdb; pdb.set_trace()
            if do_mean_var_prediction:
                t1 = time.time()
                sp_gp_mean, sp_gp_var= ss.mean_var_calc(block_size, y_data, Ait, Qi, GtY, G, GtG, H, noise_var, 
                    Kip, matrix_blocks, matrix_blocks_derivatives=None,
                    inv_precomputed=None)
                total_timings[3].append(time.time() - t1)
                    
                    
            del Ait, Qi, GtY, G, GtG, Ki_derivatives, Kip, matrix_blocks, matrix_blocks_derivatives, res
            
        if run_gp:
            print('Regular GP run:')
            t1 = time.time()
            gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
            marginal_ll_gp = float(-gp_reg.objective_function())
            
            mll_diff = np.abs( marginal_ll - marginal_ll_gp )
            mll_diff_relative =  mll_diff / np.abs(marginal_ll_gp)           
            
            reg_d_mll = -gp_reg.objective_function_gradients().copy(); reg_d_mll.shape = (reg_d_mll.shape[0],1)            
            d_mll_diff = d_marginal_ll - (reg_d_mll)
            d_mll_diff_relative = np.sum( np.abs(d_mll_diff) ) / np.sum( np.abs(reg_d_mll) ) 
            total_timings[2].append(time.time() - t1)
            
            measurements['mll_diff'] = mll_diff
            measurements['mll_diff_relative'] = mll_diff_relative
            measurements['d_mll_diff'] = d_mll_diff
            measurements['d_mll_diff_relative'] = d_mll_diff_relative
            
            
            if do_mean_var_prediction:
                t1 = time.time()
                gp_reg_mean, gp_reg_var = gp_reg.predict(x_data,include_likelihood=False)
                total_timings[4].append(time.time() - t1) 
                #import pdb; pdb.set_trace()
                mean_diff_max = np.max( np.abs( sp_gp_mean - gp_reg_mean))
                mean_diff_max_rel = np.max(np.abs( sp_gp_mean - gp_reg_mean) / np.abs(gp_reg_mean) )
                
                var_diff_max = np.max( np.abs( sp_gp_var - gp_reg_var) )
                var_diff_max_rel = np.max( np.abs( sp_gp_var - gp_reg_var) / np.abs( gp_reg_var) )
                                
                
                mean_diff_mean =  np.mean( np.abs( sp_gp_mean - gp_reg_mean))
                mean_diff_mean_rel = np.mean( np.abs( sp_gp_mean - gp_reg_mean)) / np.mean( np.abs(gp_reg_mean) )
                var_diff_mean = np.mean( np.abs( sp_gp_var - gp_reg_var))
                var_diff_mean_rel = np.mean( np.abs( sp_gp_var - gp_reg_var)) / np.mean( np.abs(gp_reg_var) )
                                
                measurements['mean_diff_max'] = mean_diff_max
                measurements['mean_diff_max_rel'] = mean_diff_max_rel
                measurements['var_diff_max'] = var_diff_max
                measurements['var_diff_max_rel'] = var_diff_max_rel
                measurements['mean_diff_mean'] = mean_diff_mean
                measurements['mean_diff_mean_rel'] = mean_diff_mean_rel
                measurements['var_diff_mean'] = var_diff_mean
                measurements['var_diff_mean_rel'] = var_diff_mean_rel
                
            del gp_reg
        
        print('Repetition no %i' % rr)

    def time_mean(i):
        """
        Means of timings over different iterations        
        
        Input:
        -------------
        
        i - index of interest
        """
        mean = 0
        for rr in range(0, repetitions_num):
            if isinstance( timings_list[rr][i], Iterable):
                mean += np.array( timings_list[rr][i] ).mean()
            else:
                mean += timings_list[rr][i]
        mean = mean / repetitions_num
        return mean

    def time_std(i):
        """
        Stds of timings over different iterations        
        
        Input:
        -------------
        
        i - index of interest
        """
        
        mean = time_mean(i)
        std = 0
        num = 0
        for rr in range(0, repetitions_num):
            if isinstance( timings_list[rr][i], Iterable):
                for val in timings_list[rr][i]:
                    std += (val - mean)**2
                    num += 1
            else:
                std += (timings_list[rr][i] - mean)**2
                num += 1
        std = np.sqrt( std / (num-1) )
        return std
    
    total_timings[0] = np.mean( total_timings[0] ) # Sparse GP initial covariance calculation
    total_timings[1] = np.mean( total_timings[1] ) # Sparse GP mll and d_mll calculation
    total_timings[2] = np.mean( total_timings[2] ) # Reg GP mll and d_mll calculation
    return block_size, measurements, timings_list, timings_descr, total_timings, sparsity_info, \
        sparsity_descr, time_mean, time_std

def scaling_measurement(result_file_name, each_size_rep_num, data_sizes=None, kernel_no=0):
    """
    Test scaling of the algorithm.
    
    Input:
    ------------------------
    result_file_name: text
    
    each_size_rep_num: int
        How much time each size computation is repeated.
        
    kernel_no: int
        Which kernel to use. Kernels are defined in test_sparse_gp_timings
    
    """
    global do_mean_var_prediction
    global run_gp
    
    if data_sizes is None:
        data_sizes = (100,500,1000,3000,5000,7000,9000,10000)
    #data_sizes = (100,500)
    
    total_time_sparse = []
    total_time_reg_gp = []
    mll_diff_list = []
    mll_diff_list_relative = []
    d_mll_diff_list = []
    d_mll_diff_list_relative = []
    
    if do_mean_var_prediction:
        mean_diff_max_list = []
        mean_diff_max_rel_list = []
        var_diff_max_list = []
        var_diff_max_rel_list = []
        mean_diff_mean_list = []
        mean_diff_mean_rel_list = []
        var_diff_mean_list = []
        var_diff_mean_rel_list = []   
        
    for ds in data_sizes:
        block_size, accuracy_measurements, timings_list, timings_descr, total_timings, sparsity_info, \
        sparsity_descr, time_mean, time_std = test_sparse_gp_timings(ds, each_size_rep_num, kernel_num=kernel_no)
        
        total_time_sparse.append( total_timings[0] + total_timings[1] )
        if run_gp:
            mll_diff = accuracy_measurements['mll_diff']
            mll_diff_relative = accuracy_measurements['mll_diff_relative']
            d_mll_diff = accuracy_measurements['d_mll_diff']
            d_mll_diff_relative = accuracy_measurements['d_mll_diff_relative']        
            
            
            total_time_reg_gp.append(total_timings[2])
            mll_diff_list.append( np.abs(mll_diff) if mll_diff is not None else np.nan )
            mll_diff_list_relative.append( mll_diff_relative if mll_diff_relative is not None else np.nan )
            d_mll_diff_list.append( np.sum(np.abs(d_mll_diff)) if d_mll_diff is not None else np.nan)
            d_mll_diff_list_relative.append( d_mll_diff_relative if d_mll_diff_relative is not None else np.nan  )
            
        
        
        if do_mean_var_prediction:
            mean_diff_max = accuracy_measurements['mean_diff_max']
            mean_diff_max_rel = accuracy_measurements['mean_diff_max_rel']
            var_diff_max = accuracy_measurements['var_diff_max']
            var_diff_max_rel = accuracy_measurements['var_diff_max_rel']
            mean_diff_mean = accuracy_measurements['mean_diff_mean']
            mean_diff_mean_rel = accuracy_measurements['mean_diff_mean_rel']
            var_diff_mean = accuracy_measurements['var_diff_mean']
            var_diff_mean_rel = accuracy_measurements['var_diff_mean_rel']   
             
            mean_diff_max_list.append( mean_diff_max if mean_diff_max is not None else np.nan )
            mean_diff_max_rel_list.append( mean_diff_max_rel if mean_diff_max_rel is not None else np.nan )
            var_diff_max_list.append( var_diff_max if var_diff_max is not None else np.nan )
            var_diff_max_rel_list.append( var_diff_max_rel if var_diff_max_rel is not None else np.nan )
            mean_diff_mean_list.append( mean_diff_mean if mean_diff_mean is not None else np.nan )
            mean_diff_mean_rel_list.append( mean_diff_mean_rel if mean_diff_mean_rel is not None else np.nan )
            var_diff_mean_list.append( var_diff_mean if var_diff_mean is not None else np.nan )
            var_diff_mean_rel_list.append( var_diff_mean_rel if var_diff_mean_rel is not None else np.nan )
             
    import scipy.io as io
    result_dict = {}
    
    result_dict['data_sizes'] = data_sizes
    result_dict['total_time_sparse'] =  total_time_sparse   
    result_dict['total_time_reg_gp'] = total_time_reg_gp
    result_dict['mll_diff_list'] = mll_diff_list
    result_dict['d_mll_diff_list'] = d_mll_diff_list
    result_dict['mll_diff_list_relative'] = mll_diff_list_relative
    result_dict['d_mll_diff_list_relative'] = d_mll_diff_list_relative
    
    if do_mean_var_prediction:
        result_dict['mean_diff_max_list'] = mean_diff_max_list
        result_dict['mean_diff_max_rel_list'] =  mean_diff_max_rel_list   
        result_dict['var_diff_max_list'] = var_diff_max_list
        result_dict['var_diff_max_rel_list'] = var_diff_max_rel_list
        result_dict['mean_diff_mean_list'] = mean_diff_mean_list
        result_dict['mean_diff_mean_rel_list'] = mean_diff_mean_rel_list
        result_dict['var_diff_mean_list'] = var_diff_mean_list
        result_dict['var_diff_mean_rel_list'] = var_diff_mean_rel_list
        
    io.savemat(result_file_name, result_dict)
    
def plot_scaling_measurements(file_name):
    
    import matplotlib.pyplot as plt
    import scipy.io as io
    
    result_dict = io.loadmat(file_name)
    
    data_sizes = result_dict['data_sizes'].squeeze()
    total_time_sparse = result_dict['total_time_sparse'].squeeze()
    total_time_reg_gp = result_dict['total_time_reg_gp'].squeeze()
    mll_diff_list = result_dict['mll_diff_list'].squeeze()
    d_mll_diff_list = result_dict['d_mll_diff_list'].squeeze()
    mll_diff_list_relative = result_dict['mll_diff_list_relative'].squeeze() 
    d_mll_diff_list_relative = result_dict['d_mll_diff_list_relative'].squeeze()
    
    plt.figure(1)
    plt.title('Running Times Comparison' )    
    plt.plot( data_sizes, total_time_reg_gp, 'bo-', label='reg gp')
    plt.plot( data_sizes, total_time_sparse, 'ro-', label='sparse gp')        
    plt.xlabel('Sample Length')
    plt.ylabel('Time (Seconds)')
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.title('Mll and d_Mll discrepancies')    
    plt.plot( data_sizes, mll_diff_list, 'bo-', label='mll (abs)')
    plt.plot( data_sizes, d_mll_diff_list, 'ro-', label='d_mll (L1)')        
    plt.xlabel('Sample Length')
    plt.ylabel('Abs or L1 norm')
    plt.legend()
    plt.show()
    
    plt.figure(3)
    plt.title('Mll and d_Mll RELATIVE discrepancies')    
    plt.plot( data_sizes, mll_diff_list_relative, 'bo-', label='mll diff relative')
    plt.plot( data_sizes, d_mll_diff_list_relative, 'ro-', label='d_mll (L1) diff relative')        
    plt.xlabel('Sample Length')
    plt.ylabel('Ration of L1 norms')
    plt.legend()
    plt.show()

def experiment_1():
    """
    Test scaling of sparse GP with comparison to regular GP.
    After code optimization.    
    
    """
    global do_mean_var_prediction
    do_mean_var_prediction = False
    #!!! Set run_gp = True in test_sparse_gp_timings 
    
    
    data_sizes = (100,500,1000,3000,5000,7000,9000,10000)
  
    # After 10000 memory swaping becomes obvioous
    
    scaling_measurement('ex1_K_0',3, data_sizes, kernel_no=0) # Metern3/2 kernel, 5 repetitions. Block size 2.
    #plot_scaling_measurements('ex1_K_0')

    scaling_measurement('ex1_K_3',3, data_sizes, kernel_no=3) # RBF kernel, 5 repetitions. Block size 10.
    #plot_scaling_measurements('ex1_K_3')

    scaling_measurement('ex1_K_4',3, data_sizes, kernel_no=4) # STD_per kernel, 5 repetitions. Block size 16.
    #plot_scaling_measurements('ex1_K_4')

def load_ex_1(file_name):
    """
    """
    import scipy.io as io
    
    result_dict = io.loadmat(file_name)
    
    data_sizes = result_dict['data_sizes'].squeeze()
    total_time_sparse = result_dict['total_time_sparse'].squeeze()
    total_time_reg_gp = result_dict['total_time_reg_gp'].squeeze()
    mll_diff_list = result_dict['mll_diff_list'].squeeze()
    d_mll_diff_list = result_dict['d_mll_diff_list'].squeeze()
    mll_diff_list_relative = result_dict['mll_diff_list_relative'].squeeze() 
    d_mll_diff_list_relative = result_dict['d_mll_diff_list_relative'].squeeze()
    
    return data_sizes, total_time_sparse, total_time_reg_gp
    
def plot_experiment_1():
    """
    """
    import matplotlib
    import matplotlib.pyplot as plt
    font = {'family' : 'sans',
             'weight' : 'bold',
             'size'   : 19}

    matplotlib.rc('font', **font)
    
    data_sizes, tot_time_sp_m32, tot_time_reg_m32 = load_ex_1('ex1_K_0')
    _, tot_time_sp_rbf, tot_time_reg_rbf = load_ex_1('ex1_K_3')
    _, tot_time_sp_per, tot_time_reg_per = load_ex_1('ex1_K_4')

    plt.figure(1)
    plt.title('Running Times Comparison', fontsize=30 )    
    plt.plot( data_sizes, tot_time_reg_m32, 'bo-', label='Standard GP (Mat32)', linewidth=3, markersize=10)
    #plt.plot( data_sizes, tot_time_reg_rbf, 'mo-', label='reg gp rbf')
    #plt.plot( data_sizes, tot_time_reg_per, 'ko-', label='reg gp per') 
    plt.plot( data_sizes, tot_time_sp_m32, 'rs--', label='Sparse Pres. Mat32', linewidth=5, markersize=10)
    plt.plot( data_sizes, tot_time_sp_rbf, 'yD-.', label='Sparse Pres. RBF', linewidth=5, markersize=10)
    plt.plot( data_sizes, tot_time_sp_per, 'gv:', label='Sparse Pres. Periodic', linewidth=5, markersize=10)
       
    plt.xlabel('Sample Length', fontsize=25)
    plt.ylabel('Time (Seconds)', fontsize=25)
    plt.legend(loc=2)
    plt.show()

def experiment_2():
    """
    Test scaling of Matern32 (block_size 2) of sparse GP.
    """
    #!!! Set run_gp = False in test_sparse_gp_timings
    global do_mean_var_prediction, run_gp
    do_mean_var_prediction = False
    run_gp = False
    
    data_sizes = (1000,5000,10000,20000,30000,40000,50000)
    #data_sizes = (1000,2000,2100)
    # After 10000 memory swaping becomes obvioous
    
    scaling_measurement('ex2_mat32',3, data_sizes)
    #plot_scaling_measurements('ex2_mat32')
def load_ex_2(file_name):
    """
    """
    import scipy.io as io
    
    result_dict = io.loadmat(file_name)
    
    data_sizes = result_dict['data_sizes'].squeeze()
    total_time_sparse = result_dict['total_time_sparse'].squeeze()
    
    return data_sizes, total_time_sparse
    
def plot_experiment_2():
    """
    """
    import matplotlib
    import matplotlib.pyplot as plt
    font = {'family' : 'sans',
             'weight' : 'bold',
             'size'   : 19}

    matplotlib.rc('font', **font)
    
    data_sizes, tot_time_sp_m32 = load_ex_2('ex2_mat32')
    
    plt.figure(1)
    plt.title('Sparse Precision GP Scaling', fontsize=30 )    
    plt.plot( data_sizes, tot_time_sp_m32, 'rs--', label='Sparse Pres. Mat32', linewidth=3, markersize=8)
    
    plt.xlabel('Sample Length', fontsize=25)
    plt.ylabel('Time (Seconds)', fontsize=25)
    plt.legend(loc=2)
    plt.show()
    
def experiment_3(plot_only=False):
    """
    Test scaling of sparce GP wrt block sizes (different kernels)
    """   
    #!!! Set run_gp = True in test_sparse_gp_timings. We want to compare accuracy with
    # regular GP.
    import scipy.io as io
    
    global do_mean_var_prediction, run_gp
    do_mean_var_prediction = False
    run_gp = True
    
    result_file_name = 'ex3'
    
    kernel_nums = (0,1,2,3,4,5,6,7)
    #kernel_nums = (0,1,3)
    
    if not plot_only:
        block_size_list = []
        mll_diff_list = []
        d_mll_diff_list = []
        d_mll_diff_relative_list = []
        total_timings_list = []
        for kk in kernel_nums:
            block_size, accuracy_measurements, timings_list, timings_descr, total_timings, sparsity_info, \
            sparsity_descr, time_mean, time_std = test_sparse_gp_timings(1000, 3, kernel_num = kk)
            
            mll_diff = accuracy_measurements['mll_diff']
            mll_diff_relative = accuracy_measurements['mll_diff_relative']
            d_mll_diff = accuracy_measurements['d_mll_diff']
            d_mll_diff_relative = accuracy_measurements['d_mll_diff_relative'] 
        
            block_size_list.append( block_size )
            mll_diff_list.append( mll_diff[0,0] )
            d_mll_diff_list.append( d_mll_diff )
            d_mll_diff_relative_list.append( d_mll_diff_relative )
            total_timings_list.append([total_timings[kk] for kk in [0,1,2] ] )
            #import pdb; pdb.set_trace()
            
        result_dict = {}
        #import pdb; pdb.set_trace()
        result_dict['block_size_list'] = block_size_list
        result_dict['mll_diff_list'] =  mll_diff_list   
        result_dict['d_mll_diff_list'] = d_mll_diff_list
        result_dict['total_timings_list'] = total_timings_list
        result_dict['d_mll_diff_relative_list'] = d_mll_diff_relative_list
    io.savemat(result_file_name, result_dict)
    return result_dict
        
def load_ex_3(file_name):
    """
    """
    import scipy.io as io
    
    result_dict = io.loadmat(file_name)
    
    block_size_list = result_dict['block_size_list'].squeeze()
    total_timings_list = result_dict['total_timings_list'].squeeze()
    #import pdb; pdb.set_trace()
    cov_matrs_time_iter = [total_timings_list[i,0] for i in range(0, total_timings_list.shape[0])]
    d_mll_time_iter = [total_timings_list[i,1] for i in range(0, total_timings_list.shape[0])]
    reg_gp_time_iter = [total_timings_list[i,2] for i in range(0, total_timings_list.shape[0])]
    
    return block_size_list, cov_matrs_time_iter, d_mll_time_iter, reg_gp_time_iter
    
def plot_experiment_3():
    """
    """
    import matplotlib
    import matplotlib.pyplot as plt
    font = {'family' : 'sans',
             'weight' : 'bold',
             'size'   : 19}

    matplotlib.rc('font', **font)
    
    block_size_list, cov_matrs_time_iter, d_mll_time_iter, reg_gp_time_iter = load_ex_3('ex3')
    
    plt.figure(1)
    plt.title('Running Time vs Block_size', fontsize=30)    
    plt.plot( block_size_list, cov_matrs_time_iter, 'ms--', label='Sparse Pres. (cov_matrs)',linewidth=5, markersize=10)
    plt.plot( block_size_list, d_mll_time_iter, 'rv-.', label='Sparse Pres. (d_MLL)',linewidth=5, markersize=10)
    plt.plot( block_size_list, reg_gp_time_iter, 'bo-', label='Standard GP', linewidth=3, markersize=10)        
    plt.xlabel('Block_size', fontsize=25)
    plt.ylabel('Time (Seconds)', fontsize=25)
    plt.legend(loc=2)
    plt.show()
    
    #    d_mll_l1_relative_iter = [d_mll_diff_relative_list[i] for i in range(0, len(total_timings_list))]
    #    d_mll_l1_iter = [np.sum(np.abs(d_mll_diff_list[i])) for i in range(0, len(d_mll_diff_list))]
    #    
    #    plt.figure(2)
    #    plt.title('Mll discrepancies. Samples 1000')    
    #    plt.plot( block_size_list, np.abs(mll_diff_list), 'bo-', label='mll (abs)')       
    #    plt.xlabel('Block_sizes')
    #    plt.ylabel('Abs')
    #    plt.legend()
    #    plt.show()
    #    
    #    plt.figure(3)
    #    plt.title('d_Mll discrepancies. Samples 1000')
    #    plt.plot( block_size_list, d_mll_l1_iter, 'ro-', label='d_mll (L1)')        
    #    plt.xlabel('Block_sizes')
    #    plt.ylabel('L1 norm')
    #    plt.legend()
    #    plt.show()
    #    
    #    plt.figure(4)
    #    plt.title('d_Mll relative discrepancies. Samples 1000')
    #    plt.plot( block_size_list, d_mll_l1_relative_iter, 'ro-', label='d_mll (L1)')        
    #    plt.xlabel('Block_sizes')
    #    plt.ylabel('L1 norm / L1 norm')
    #    plt.legend()
    #    plt.show()

def experiment_4():
    """
    Test scaling of sparse GP with comparison to regular GP.
    After code optimization.    
    
    """
    global do_mean_var_prediction
    do_mean_var_prediction = True
    #!!! Set run_gp = True in test_sparse_gp_timings 
    
    
    data_sizes = (100,500,1000,2500, 5000)
  
    # After 10000 memory swaping becomes obvioous
    
    scaling_measurement('ex4_K_0',3, data_sizes, kernel_no=0) # Metern3/2 kernel, 5 repetitions. Block size 2.
    #plot_scaling_measurements('ex1_K_0')

    scaling_measurement('ex4_K_3',3, data_sizes, kernel_no=3) # RBF kernel, 5 repetitions. Block size 10.
    #plot_scaling_measurements('ex1_K_3')

    scaling_measurement('ex4_K_4',3, data_sizes, kernel_no=4) # STD_per kernel, 5 repetitions. Block size 16.
    #plot_scaling_measurements('ex1_K_4')
def load_ex_4(file_name):
    """
    """
    import scipy.io as io
    
    result_dict = io.loadmat(file_name)
    
    data_sizes = result_dict['data_sizes'].squeeze()
    total_time_sparse = result_dict['total_time_sparse'].squeeze()
    total_time_reg_gp = result_dict['total_time_reg_gp'].squeeze()
    mll_diff_list = result_dict['mll_diff_list'].squeeze()
    d_mll_diff_list = result_dict['d_mll_diff_list'].squeeze()
    mll_diff_list_relative = result_dict['mll_diff_list_relative'].squeeze() 
    d_mll_diff_list_relative = result_dict['d_mll_diff_list_relative'].squeeze()
    
    mean_diff_max_list = result_dict['mean_diff_max_list'].squeeze()
    mean_diff_max_rel_list = result_dict['mean_diff_max_rel_list'].squeeze()
    var_diff_max_list  =  result_dict['var_diff_max_list'].squeeze()
    var_diff_max_rel_list = result_dict['var_diff_max_rel_list'].squeeze()
    mean_diff_mean_list = result_dict['mean_diff_mean_list'].squeeze()
    mean_diff_mean_rel_list = result_dict['mean_diff_mean_rel_list'].squeeze()
    var_diff_mean_list = result_dict['var_diff_mean_list'].squeeze()
    var_diff_mean_rel_list = result_dict['var_diff_mean_rel_list'].squeeze()
        
    return data_sizes, mll_diff_list, d_mll_diff_list, mll_diff_list_relative, d_mll_diff_list_relative, \
        mean_diff_max_list, mean_diff_max_rel_list, var_diff_max_list, var_diff_max_rel_list, \
        mean_diff_mean_list, mean_diff_mean_rel_list, var_diff_mean_list, var_diff_mean_rel_list

def process_experiment_4(file_name):
    """
    """
    res = load_ex_4(file_name)      
    return res
    
def experiment_5(action =2):
    """
    !!! for this experiment we need to make regularization jitter equal 1e-8.
        And add artificial regularization into state-space main.
    """
    
    import GPy.models.state_space_model as ss_model
    import GPy.models.ss_sparse_model as ss_sparse_model
    # Data ->
    data_file_path = '/home/alex/Programming/python/Sparse GP/d2_shorter_clean2.csv'    
    
    y_data = np.loadtxt(data_file_path); y_data.shape = (y_data.shape[0],1)
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = np.arange(0, y_data.shape[0]); x_data.shape = (x_data.shape[0],1)
    # Data <-
    
    # Find hyper parameters -> 
    
    var_1_day = 1.0
    ls_1_day = 100
    var_2_week = 1.0
    ls_2_week = 100
    var_3_quasi = 1.0
    ls_3_quasi = 300
    var_4_year = 1.0
    ls_4_year = 100
    
    noise_var = 0.34
#    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_day, period = 24, lengthscale=ls_1_day) +\
#              GPy.kern.sde_StdPeriodic(1, variance=var_2_week, period = 168, lengthscale=ls_2_week) +\
#              GPy.kern.sde_RBF(1, variance=var_3_quasi, lengthscale=ls_3_quasi)*\
#              GPy.kern.sde_StdPeriodic(1, variance=var_4_year, period = 8640, lengthscale=ls_4_year)
#              
#    kernel1.std_periodic.variance.constrain_positive()
#    kernel1.std_periodic.period.fix()
#    kernel1.std_periodic_1.period.fix()
#    kernel1.mul.std_periodic.period.fix()
#    
#    kernel2 = GPy.kern.StdPeriodic(1, variance=var_1_day, period = 24, lengthscale=ls_1_day) +\
#              GPy.kern.StdPeriodic(1, variance=var_2_week, period = 168, lengthscale=ls_2_week) +\
#              GPy.kern.RBF(1, variance=var_3_quasi, lengthscale=ls_3_quasi)*\
#              GPy.kern.StdPeriodic(1, variance=var_4_year, period = 8640, lengthscale=ls_4_year)
#              
#    kernel2.std_periodic.variance.constrain_positive()
#    kernel2.std_periodic.period.fix()
#    kernel2.std_periodic_1.period.fix()
#    kernel2.mul.std_periodic.period.fix()
    kernel1 = \
              GPy.kern.sde_StdPeriodic(1, variance=var_2_week, period = 168, lengthscale=ls_2_week) +\
              GPy.kern.sde_StdPeriodic(1, variance=var_4_year, period = 8640, lengthscale=ls_4_year)
    #import pdb; pdb.set_trace()
    kernel1.std_periodic.variance.constrain_positive()
    kernel1.std_periodic.period.fix()
    kernel1.std_periodic_1.variance.constrain_positive()
    kernel1.std_periodic_1.period.fix()
    
    #kernel2 = GPy.kern.StdPeriodic(1, variance=var_2_week, period = 168, lengthscale=ls_2_week) +\
    #kernel2= #GPy.kern.StdPeriodic(1, variance=var_1_day, period = 24, lengthscale=ls_1_day) +\
    kernel2 = GPy.kern.StdPeriodic(1, variance=var_2_week, period = 168, lengthscale=ls_2_week) +\
              GPy.kern.sde_StdPeriodic(1, variance=var_4_year, period = 8640, lengthscale=ls_4_year)
    kernel2.std_periodic.variance.constrain_positive()
    kernel2.std_periodic.period.fix()
    kernel2.std_periodic_1.variance.constrain_positive()
    kernel2.std_periodic_1.period.fix()         
    
    
    #import pdb; pdb.set_trace()
    #sparse_model = ss_sparse_model.SparcePrecisionGP(x_data, y_data, kernel1, noise_var=noise_var )
    #ssm = ss_model.StateSpace(x_data,y_data,kernel1, noise_var=noise_var,kalman_filter_type = 'svd' )
    
    if (action == 1): # train hyperparameters
        y_data = y_data[0:1000,:]; x_data = x_data[0:1000,:]   
        gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
        gp_reg.Gaussian_noise.variance.fix()
        
        gp_reg.optimize(messages=True)
        print(gp_reg)
        
        import scipy.io as io
        result_dict = {}
        result_dict['params'] = gp_reg.param_array[:]
        io.savemat('ex5_params', result_dict)
    elif (action == 2):
        y_data = y_data[0:1000,:]; x_data = x_data[0:1000,:]   
        gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
        
        import scipy.io as io
        result_dict = io.loadmat('ex5_params')
        params = result_dict['params']
        gp_reg[:] = params
        print(gp_reg)
        print(params)
        gp_mean, gp_var = gp_reg.predict(x_data)
        
        #import pdb; pdb.set_trace()
        
        import matplotlib
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.title('Electricity Consumption Data', fontsize=30)    
        plt.plot( x_data, y_data, 'g-', label='Data',linewidth=1, markersize=5)
        plt.plot( x_data, gp_mean, 'b-', label='Data',linewidth=1, markersize=5)
        plt.plot( x_data, gp_mean+gp_var, 'r--', label='Data',linewidth=1, markersize=5)
        plt.plot( x_data, gp_mean-gp_var, 'r--', label='Data',linewidth=1, markersize=5)
        plt.xlabel('Time (Hours)', fontsize=25)
        plt.ylabel('Normalized Value', fontsize=25)
        #plt.legend(loc=2)
        plt.show()
    elif (action == 3):
        
        import scipy.io as io
        result_dict = io.loadmat('ex5_params')
        params = result_dict['params'].squeeze()
        
        noise_var = params[-1]
#        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=params[0], period = 24, lengthscale=params[1]) +\
#              GPy.kern.sde_StdPeriodic(1, variance=params[2], period = 168, lengthscale=params[3]) +\
#              GPy.kern.sde_Exponential(1, variance=params[4], lengthscale=params[5])*\
#              GPy.kern.sde_StdPeriodic(1, variance=params[6], period = 8640, lengthscale=params[7])
#        
        kernel1 = \
              GPy.kern.sde_StdPeriodic(1, variance=params[0], period = 168, lengthscale=params[2]) +\
              GPy.kern.sde_StdPeriodic(1, variance=params[3], period = 8640, lengthscale=params[5])
        #import pdb; pdb.set_trace()
        #sparse_model = ss_sparse_model.SparcePrecisionGP(x_data, y_data, kernel1, noise_var=noise_var )
        
        (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
        block_size = F.shape[1]    
        
        grad_calc_params = {}
        grad_calc_params['dP_inf'] = dP_inft
        grad_calc_params['dF'] = dFt
        grad_calc_params['dQc'] = dQct
        #import pdb;pdb.set_trace()
    
        t1 = time.time()
        (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
         matrix_blocks_derivatives) = ss.sparse_inverse_cov(x_data, 
                y_data, F, L, Qc, P_inf, P0, H, compute_derivatives=True,
                           grad_calc_params=grad_calc_params)        
        
        sp_gp_mean, sp_gp_var= ss.mean_var_calc(block_size, y_data, Ait, Qi, GtY, G, GtG, H, noise_var, 
                    Kip, matrix_blocks, matrix_blocks_derivatives=None,
                    inv_precomputed=None)
        t1 = ( time.time() - t1)
        print("Sparse GP time: %f" % (t1,) )
        
        result_dict = {}
        result_dict['t1'] = t1
        result_dict['sp_gp_mean'] = sp_gp_mean
        result_dict['sp_gp_var'] = sp_gp_var + noise_var
        io.savemat('ex5_sparse_pred', result_dict)
        #sparse_mll = float(-gp_reg.objective_function())
        #sparse_d_mll = -gp_reg.objective_function_gradients().copy(); sparse_d_mll.shape = (sparse_d_mll.shape[0],1)            
        #print(sparse_mll)
        #print(sparse_d_mll)
    elif (action == 4):
        import scipy.io as io
        result_dict = io.loadmat('ex5_params')
        params = result_dict['params'].squeeze()
        
        noise_var = params[-1]
#        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=params[0], period = 24, lengthscale=params[1]) +\
#              GPy.kern.sde_StdPeriodic(1, variance=params[2], period = 168, lengthscale=params[3]) +\
#              GPy.kern.sde_Exponential(1, variance=params[4], lengthscale=params[5])*\
#              GPy.kern.sde_StdPeriodic(1, variance=params[6], period = 8640, lengthscale=params[7])
        t1 = time.time()
        kernel1 = \
              GPy.kern.sde_StdPeriodic(1, variance=params[0], period = 168, lengthscale=params[2]) +\
              GPy.kern.sde_StdPeriodic(1, variance=params[3], period = 8640, lengthscale=params[5])
        
        ssm = ss_model.StateSpace(x_data,y_data,kernel1, noise_var=noise_var,kalman_filter_type = 'svd' )
         
        ssm_mean, ssm_var = ssm.predict(x_data)
        t1 = (time.time() - t1)
        print("Ssm GP time: %f" % (t1,) )
         
        result_dict = {}
        result_dict['t1'] = t1
        result_dict['ssm_mean'] = ssm_mean
        result_dict['ssm_var'] = ssm_var
        io.savemat('ex5_ssm_pred', result_dict)
         
        #print(sparse_model)
    #sparse_model.optimize(messages=True)
    #ssm.optimize(messages=True)
    # Find hyper parameters <-
    #kernel1.constrain_fixed()
    
def plot_experiment_5(what_plot = 1):
    """
    """
    import scipy.io as io
    data_file_path = '/home/alex/Programming/python/Sparse GP/d2_shorter_clean2.csv'    
    
    y_data = np.loadtxt(data_file_path); y_data.shape = (y_data.shape[0],1)
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = np.arange(0, y_data.shape[0]); x_data.shape = (x_data.shape[0],1)
    
    result_dict = io.loadmat('ex5_ssm_pred')
    t1_ssm = result_dict['t1']
    ssm_mean = result_dict['ssm_mean']
    ssm_var = result_dict['ssm_var']
    
    result_dict = io.loadmat('ex5_sparse_pred')
    t1_sparse = result_dict['t1']
    sp_gp_mean = result_dict['sp_gp_mean']
    sp_gp_var = result_dict['sp_gp_var']
    
    max_mean_diff_rel = np.mean(np.abs(sp_gp_mean - ssm_mean) / np.abs( ssm_mean))
    max_var_diff_rel = np.mean(np.abs(sp_gp_var - ssm_var) / np.abs( ssm_var))
    print("Mean rel. mean: %f" % (max_mean_diff_rel,))    
    print("Mean rel. var: %f" % (max_var_diff_rel,)) 
    print("Time ssm: %f" % (t1_ssm, ) )    
    print("Time sparse: %f" % (t1_sparse, ) ) 
    
    import matplotlib
    import matplotlib.pyplot as plt
    font = {'family' : 'sans',
             'weight' : 'bold',
             'size'   : 19}
    matplotlib.rc('font', **font)
    
    
    #import pdb;pdb.set_trace()
    ax1 = plt.figure(1)
    #plt.title('Electricity Consumption Modeling', fontsize=30)
    plt.title('Electricity Consumption Data', fontsize=30)
    plt.plot( x_data, y_data, 'g-', label='Data',linewidth=1, markersize=1)
    #plt.plot( x_data, y_data, 'go-', label='Data',linewidth=1, markersize=3)
    #plt.plot( x_data, ssm_mean, 'bs-', label='GP-SSM',linewidth=1, markersize=3)
    #plt.plot( x_data, sp_gp_mean, 'rv-', label='Sparse Pres.',linewidth=1, markersize=3)
    #plt.plot( x_data, ssm_mean + ssm_var, 'b--', label='Data',linewidth=1, markersize=3)
    #plt.plot( x_data, sp_gp_mean + sp_gp_var, 'r--', label='Data',linewidth=1, markersize=3)
    plt.xlabel('Time (Hours)', fontsize=25)
    plt.ylabel('Normalized Value', fontsize=25)
    plt.legend(loc=2)
    ax = plt.gca()
    ax.set_xticks((0,4000, 8000, 12000, 16000))
    plt.show()
    
def experiment_6(action =1):
    """
    """
    import os
    import GPy.models.state_space_model as ss_model
    import GPy.models.ss_sparse_model as ss_sparse_model
    # Data ->
    #data_file_path = '/home/alex/Programming/python/Sparse GP/co2_weekly_clean.csv'    
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
    #data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_mlo.txt'
    
    results_filex_prefix = '/home/agrigori/Programming/python/Sparse GP/Experiemnts/Results'
    
    #import pdb; pdb.set_trace()
    data = np.loadtxt(data_file_path); 
    data = data[ np.where(data[:,1] > 0)[0] ,:] # get rid of missing values
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = x_data - 1974
    # Data <-
    
    # Find hyper parameters -> 
    var_1_trend = 1.0
    ls_1_trend = 200
    
    per_per = 1
    per_var = 1.0 # fixed
    per_ls = 1
    var_2_quasi = 1.0
    ls_2_quasi = 1.0
    
    var_3_quasi = 1.0
    ls_3_quasi = 100.0
    
    noise_var = 0.1
    #import pdb; pdb.set_trace()
    kernel1 = GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend) +\
              GPy.kern.sde_StdPeriodic(1, variance=per_var, period = per_per, lengthscale=per_ls)*\
              GPy. kern.sde_Matern32(1, variance=var_2_quasi, lengthscale=ls_2_quasi) +\
              GPy. kern.sde_Matern32(1, variance=var_3_quasi, lengthscale=ls_3_quasi)
    
    #import pdb; pdb.set_trace()
    #kernel1.std_periodic.variance.constrain_positive()
    kernel1.mul.std_periodic.variance.fix()
    kernel1.mul.std_periodic.period.fix()
    #kernel1.std_periodic_1.period.fix()
    #kernel1.mul.std_periodic.period.fix() 
#    
#    kernel2 = GPy.kern.StdPeriodic(1, variance=var_1_day, period = 24, lengthscale=ls_1_day) +\
#              GPy.kern.StdPeriodic(1, variance=var_2_week, period = 168, lengthscale=ls_2_week) +\
#              GPy.kern.RBF(1, variance=var_3_quasi, lengthscale=ls_3_quasi)*\
#              GPy.kern.StdPeriodic(1, variance=var_4_year, period = 8640, lengthscale=ls_4_year)

    kernel2 = GPy.kern.RBF(1, variance=var_1_trend, lengthscale=ls_1_trend) +\
              GPy.kern.StdPeriodic(1, variance=per_var, period = per_per, lengthscale=per_ls)*\
              GPy.kern.Matern32(1, variance=var_2_quasi, lengthscale=ls_2_quasi) +\
              GPy.kern.Matern32(1, variance=var_3_quasi, lengthscale=ls_3_quasi)
    
    kernel2.mul.std_periodic.variance.fix()
    kernel2.mul.std_periodic.period.fix()
    
#    kernel2.std_periodic.variance.constrain_positive()
#    kernel2.std_periodic.period.fix()
#    kernel2.std_periodic_1.period.fix()
#    kernel2.mul.std_periodic.period.fix()
     
    
    if (action == 1): # train hyperparameters
        import scipy.io as io
        y_data = y_data[0:1000,:]; x_data = x_data[0:1000,:]   
        gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
        gp_reg.optimize(messages=True)
        print(gp_reg)
        result_dict = {}
        result_dict['params'] = gp_reg.param_array[:]
        
#        ssm = ss_model.StateSpace(x_data,y_data,kernel1, noise_var=noise_var,kalman_filter_type = 'svd' )
#        ssm.optimize(messages=True)
#        print(ssm)
#        result_dict = {}
#        result_dict['params'] = ssm.param_array[:]
        
#        sparse_model = ss_sparse_model.SparcePrecisionGP(x_data, y_data, kernel1, noise_var=noise_var )
#        sparse_model.optimize(messages=True)
#        print(sparse_model)
        #result_dict = {}
        #result_dict['params'] = sparse_model.param_array[:]
        
        
        io.savemat(os.path.join(results_filex_prefix,'ex6_params'), result_dict)
    elif (action == 2): # plot trained hyperparameters
        #y_data = y_data[0:1000,:]; x_data = x_data[0:1000,:]   
        gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
        
        import scipy.io as io
        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_params'))
        #import pdb; pdb.set_trace()
        
        params = result_dict['params']
        #params[0,-2] = 30
        gp_reg[:] = params
        print(gp_reg)
        print(params)
        
        years_to_predict = 8
        step = np.mean( np.diff(x_data[:,0]) )
        
        x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
        
        x_new = np.vstack( (x_data, x_new)) # combine train and test data
        
        gp_mean, gp_var = gp_reg.predict(x_new)
        
        #import pdb; pdb.set_trace()
        
        import matplotlib
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.title('Electricity Consumption Data', fontsize=30)    
        plt.plot( x_data, y_data, 'g-', label='Data',linewidth=1, markersize=5)
        plt.plot( x_new, gp_mean, 'b-', label='Data',linewidth=1, markersize=5)
        plt.plot( x_new, gp_mean+gp_var, 'r--', label='Data',linewidth=1, markersize=5)
        plt.plot( x_new, gp_mean-gp_var, 'r--', label='Data',linewidth=1, markersize=5)
        plt.xlabel('Time (Hours)', fontsize=25)
        plt.ylabel('Normalized Value', fontsize=25)
        #plt.legend(loc=2)
        plt.show()
    elif (action == 3):
        
        import scipy.io as io
        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_params'))
        params = result_dict['params'].squeeze()
        params[-1] = 0.02
#        kernel1[:] = params[0:-1]
#        noise_var = params[-1]

        years_to_predict = 8
        step = np.mean( np.diff(x_data[:,0]) )
        
        x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
        
        x_new = np.vstack( (x_data, x_new)) # combine train and test data
        t1 = time.time()

    
        t1 = time.time()
        sparse_model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel1, noise_var=noise_var, p_Inv_jitter=1e-14 )
        sparse_model[:] = params
        print(sparse_model)
        sp_gp_mean, sp_gp_var = sparse_model.predict(Xnew=x_new,p_Inv_jitter=1e-16)
        
        import pdb;pdb.set_trace()
        t1 = ( time.time() - t1)
        print("Sparse GP time: %f" % (t1,) )
        
        result_dict = {}
        result_dict['t1'] = t1
        if x_new is not None:
            result_dict['x_new'] = x_new
        result_dict['sp_gp_mean'] = sp_gp_mean
        result_dict['sp_gp_var'] = sp_gp_var + noise_var
        io.savemat(os.path.join(results_filex_prefix,'ex6_sparse_pred'), result_dict)
        #sparse_mll = float(-gp_reg.objective_function())
        #sparse_d_mll = -gp_reg.objective_function_gradients().copy(); sparse_d_mll.shape = (sparse_d_mll.shape[0],1)            
        #print(sparse_mll)
        #print(sparse_d_mll)
    elif (action == 4):
        import scipy.io as io
        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_params'))
        params = result_dict['params'].squeeze()
        
        kernel1[:] = params[0:-1]
        noise_var = params[-1]        
        #import pdb; pdb.set_trace()
        t1 = time.time()
      
        ssm = ss_model.StateSpace(x_data,y_data,kernel1, noise_var=noise_var, kalman_filter_type = 'svd' )
        ssm[:] = params
        print(ssm)
        print(params)        
        
        years_to_predict = 8
        step = np.mean( np.diff(x_data[:,0]) )
        
        x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
        
        x_new = np.vstack( (x_data, x_new)) # combine train and test data
        t1 = time.time()
        
        ssm_mean, ssm_var = ssm.predict(x_new)
        t1 = (time.time() - t1)
        print("Ssm GP time: %f" % (t1,) )
         
        result_dict = {}
        if x_new is not None:
            result_dict['x_new'] = x_new
        result_dict['t1'] = t1
        result_dict['ssm_mean'] = ssm_mean
        result_dict['ssm_var'] = ssm_var
        io.savemat(os.path.join(results_filex_prefix,'ex6_ssm_pred'), result_dict)
         
        #print(sparse_model)
    #sparse_model.optimize(messages=True)
    #ssm.optimize(messages=True)
    # Find hyper parameters <-
    #kernel1.constrain_fixed()
    
def plot_experiment_6(what_plot = 1):
    """
    """
    import matplotlib
    import matplotlib.pyplot as plt
    font = {'family' : 'sans',
                 'weight' : 'bold',
                 'size'   : 19}
    matplotlib.rc('font', **font)    
    
    
    plot_data = False
    plot_sparse_pred = True
    plot_ssm = False
    import os # for path manipulations
    
    import scipy.io as io
    #data_file_path = '/home/alex/Programming/python/Sparse GP/co2_weekly_clean.csv'    
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/co2_weekly_clean.csv'
    
    results_filex_prefix = '/home/agrigori/Programming/python/Sparse GP/Experiemnts/Results'
    
    #import pdb; pdb.set_trace()
    data = np.loadtxt(data_file_path); 
    data = data[ np.where(data[:,1] > 0)[0] ,:]
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = x_data - 1974
    
    if plot_data:
        plt.figure(1)
        plt.title('Electricity Consumption Modeling', fontsize=30)
        plt.plot( x_data+ 1974, y_data, 'go-', label='Data',linewidth=1, markersize=3)
        #plt.plot( x_data, ssm_mean, 'bs-', label='GP-SSM',linewidth=1, markersize=3)
        #plt.plot( x_data, sp_gp_mean, 'rv-', label='Sparse Pres.',linewidth=1, markersize=3)
        #plt.plot( x_data, ssm_mean + ssm_var, 'b--', label='Data',linewidth=1, markersize=3)
        #plt.plot( x_data, sp_gp_mean + sp_gp_var, 'r--', label='Data',linewidth=1, markersize=3)
        plt.xlabel('Time (Hours)', fontsize=25)
        plt.ylabel('Normalized Value', fontsize=25)
        plt.legend(loc=2)
        plt.show()

    elif plot_sparse_pred:    
        
        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_ssm_pred'))
        t1_ssm = result_dict['t1']
        ssm_mean = result_dict['ssm_mean']
        ssm_var = result_dict['ssm_var']
        tmp = result_dict.get('x_new')
        if tmp is not None:
            x_ssm = tmp
        else:
            x_ssm = x_data
        
        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_sparse_pred'))
        t1_sparse = result_dict['t1']
        sp_gp_mean = result_dict['sp_gp_mean']
        sp_gp_var = result_dict['sp_gp_var']
        tmp = result_dict.get('x_new')
        if tmp is not None:
            x_sparse = tmp
        else:
            x_sparse = x_data
        
        
#        max_mean_diff_rel = np.mean(np.abs(sp_gp_mean - ssm_mean) / np.abs( ssm_mean))
#        max_var_diff_rel = np.mean(np.abs(sp_gp_var - ssm_var) / np.abs( ssm_var))
#        print("Mean rel. mean: %f" % (max_mean_diff_rel,))    
#        print("Mean rel. var: %f" % (max_var_diff_rel,)) 
#        print("Time ssm: %f" % (t1_ssm, ) )    
#        print("Time sparse: %f" % (t1_sparse, ) ) 
        #import pdb; pdb.set_trace()
        #import pdb;pdb.set_trace()
        plt.figure(1)
        plt.title('Electricity Consumption Modeling', fontsize=30)
        plt.plot( x_data+ 1974, y_data, 'go-', label='Data',linewidth=1, markersize=3)
        #plt.plot( x_ssm+ 1974, ssm_mean, 'bs-', label='GP-SSM',linewidth=1, markersize=3)
        plt.plot( x_sparse+ 1974, sp_gp_mean, 'r-', label='Sparse Pres.',linewidth=1, markersize=3)
        #plt.plot( x_ssm+ 1974, ssm_mean + ssm_var, 'b--', label='Data',linewidth=1, markersize=3)
        #plt.plot( x_sparse+1974, sp_gp_mean + sp_gp_var, 'r--', label='Data',linewidth=1, markersize=3)
        plt.xlabel('Time (Hours)', fontsize=25)
        plt.ylabel('Normalized Value', fontsize=25)
        #plt.ylim((-2, 1e4))
        #plt.xlim((2012,2020))
        #plt.legend(loc=2)
        plt.show()
    
if  __name__ == '__main__':
    #test_sparse_determinant_computation()
    #test_sparse_inverse_cov_simple()
    #pass
    
    
    #experiment_1()
    #plot_experiment_1()
    #experiment_2()
    #plot_experiment_2()
    #experiment_3()
    #plot_experiment_3()
    #rd = experiment_3()
    
    #experiment_5(action = 3
    #plot_experiment_5()
    #plot_experiment_5()
    #experiment_4()
    #
    #experiment_6(action=3)
    plot_experiment_6()
    # Tomorrow check variance calculation! Change sizes
#    data_sizes, mll_diff_list, d_mll_diff_list, mll_diff_list_relative, , \
#    mean_diff_max_list, mean_diff_max_rel_list, var_diff_max_list, var_diff_max_rel_list, \
#    mean_diff_mean_list, mean_diff_mean_rel_list, var_diff_mean_list, var_diff_mean_rel_list = process_experiment_4('ex4_K_4')
#    
    
    #0, 1, 2, 4 - in tests. The rest - large gradient difference
    #block_size, mll_diff , d_mll_diff, d_mll_diff_relative, timings_list, timings_descr, total_timings, sparsity_info, \
    #    sparsity_descr, time_mean, time_std = test_sparse_gp_timings(100, 1, kernel_num = 6) # 3-, 4-, 5-, 6-, 7-
    #test_matrix_cook_book_f164()
    
    #scaling_measurement('second_result')
    #plot_scaling_measurements('second_result')
    
    #constr_times, read_times = test_random_matr_slicing_speeds(10, 1000, 5)