import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scipy.optimize import minimize

import time

import sys
sys.path.append('PAL_cineca/Scripts/')
from measles_PALSMC import *
from measles_simulator import *

simulation_name = "seq_optimization_C_05_1_trance"

#############################################
# Load the data
#############################################
UKbirths_array = np.load("PAL_cineca/Data/Input/UKbirths_array.npy")
UKpop_array = np.load("PAL_cineca/Data/Input/UKpop_array.npy")
measles_distance_matrix_array = np.load("PAL_cineca/Data/Input/measles_distance_matrix_array.npy")
UKmeasles_array = np.load("PAL_cineca/Data/Input/UKmeasles_array.npy")
UKbirths = tf.convert_to_tensor(UKbirths_array, dtype = tf.float32)
UKpop = tf.convert_to_tensor(UKpop_array, dtype = tf.float32)
measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array, dtype = tf.float32)
UKmeasles = tf.convert_to_tensor(UKmeasles_array, dtype = tf.float32)

term   = tf.convert_to_tensor([6, 99, 115, 198, 252, 299, 308, 355, 366], dtype = tf.float32)
school = tf.convert_to_tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype = tf.float32)

n_cities = tf.constant(40, dtype = tf.int64)

initial_pop = UKpop[:,0]

delta_year = tf.convert_to_tensor([[1/50]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)

T = 420
intermediate_steps = 4
is_school_term_array, is_start_school_year_array, times_total, times_obs = school_term_and_school_year(T, intermediate_steps, term, school)

is_school_term_array = tf.convert_to_tensor(is_school_term_array, dtype = tf.float32)
is_start_school_year_array = tf.convert_to_tensor(is_start_school_year_array, dtype = tf.float32)

intermediate_steps = tf.constant(4, dtype = tf.float32)
h = tf.constant(14/tf.cast(intermediate_steps, dtype = tf.float32), dtype = tf.float32)

delta_year = tf.convert_to_tensor([[1/50]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)

p      = tf.constant(0.739, dtype = tf.float32)
q_mean = tf.constant(np.mean(np.load("PAL_cineca/Data/Input/q_mean.npy")), dtype = tf.float32)


string1 = ["################################", "\n"]
string2 = ["Data loaded", "\n"]
f= open("PAL_cineca/Data/Check/"+simulation_name+".txt", "a")
f.writelines(string1)
f.writelines(string2)
f.close()


#########################################################
# Optimization
#########################################################

optimal_B_param = np.load("PAL_cineca/Data/Output/seq_optimization_B_05_3_trance.npy")
x_initial = optimal_B_param[:3*n_cities]
x_beta = optimal_B_param[3*n_cities:(3*n_cities+1)]*np.ones((n_cities))
x_rest = optimal_B_param[(3*n_cities+1):]
x_to_start = np.concatenate((x_initial, x_beta, x_rest))

n_particles = 6000

counter = 0

def optimization_func(x_optim):
    
    x = np.exp(x_optim) 

    x_initial = np.reshape(x[0:n_cities.numpy()*3], (n_cities, 3))
    pi_0_init = tf.convert_to_tensor(np.concatenate((x_initial[:,0:1], 
                                                     x_initial[:,1:2]*(1-x_initial[:,0:1]), 
                             			     x_initial[:,2:3]*(1-x_initial[:,1:2])*(1-x_initial[:,0:1]), 
                             			     1 - x_initial[:,0:1] 
                             			       - x_initial[:,1:2]*(1-x_initial[:,0:1]) 
                               			       - x_initial[:,2:3]*(1-x_initial[:,1:2])*(1-x_initial[:,0:1])), 
                               			     axis = 1), dtype = np.float32)

    x_beta = x[(n_cities.numpy()*3):(n_cities.numpy()*3+n_cities.numpy())]
    beta_bar_init = tf.convert_to_tensor(np.reshape(x_beta, (n_cities, 1)), dtype = tf.float32)

    x_rest = x[(n_cities.numpy()*3+n_cities.numpy()):]
    rho_init   = tf.convert_to_tensor([x_rest[0:1]], dtype = tf.float32)
    gamma_init = tf.convert_to_tensor([x_rest[1:2]], dtype = tf.float32)
    g_init     = 100*tf.convert_to_tensor([x_rest[2:3]], dtype = tf.float32)
    a_init     = tf.convert_to_tensor( x_rest[3], dtype = tf.float32)
    c_init     = tf.convert_to_tensor( x_rest[4], dtype = tf.float32)
    xi_var     = 10*tf.convert_to_tensor( x_rest[5], dtype = tf.float32)
    q_var      = tf.convert_to_tensor( x_rest[6], dtype = tf.float32)
    
    pi_0_init_transform     = pi_0_init
    beta_bar_init_transform = beta_bar_init
    rho_init_transform      = rho_init*tf.ones((n_cities, 1), dtype = tf.float32)
    gamma_init_transform    = gamma_init*tf.ones((n_cities, 1), dtype = tf.float32)
    g_init_transform        = g_init*tf.ones((n_cities, 1), dtype = tf.float32)

    Xi = tfp.distributions.Gamma(concentration = xi_var, rate = xi_var)
    Q  = tfp.distributions.TruncatedNormal( q_mean, q_var, 0, 1)

    # The foloowing is running the lookahead, if you want to using a vanilla SMC use PAL_run_likelihood_res
    value = -PAL_SMC( intermediate_steps, UKmeasles, UKbirths, UKpop, 
                                g_init_transform, measles_distance_matrix, initial_pop, 
                   		pi_0_init_transform, beta_bar_init_transform, p, a_init, 
                		is_school_term_array, is_start_school_year_array, h, rho_init_transform, gamma_init_transform, Xi, Q, c_init, n_cities, n_particles, delta_year).numpy()
    
    global counter
    counter = counter+1
    
    if counter%1==0:
        string = ["The loglikelihood at the "+str(counter)+" evaluation is:"+str(-value), "\n"]
        string_par = ["and we are evaluating:", "\n"]
        string_par_next = [" "+str(x_optim), "\n"] 
        f= open("PAL_cineca/Data/Check/"+simulation_name+".txt", "a")
        f.writelines(string)
        f.writelines(string_par)
        f.writelines(string_par_next)
        f.close()

    return value

# Minimize the objective function subject to the constraints
bnds_x_0 = (((-20, -0.01),)*n_cities.numpy()*3)

bnds_x_0 = (((-2, -0.01),))

for i in range(1, n_cities.numpy()*3):
    
    if i%3==0:
        bnds_x_0 = bnds_x_0 + (((-2, -0.01),))
        
    else:
        bnds_x_0 = bnds_x_0 + (((-10, -0.01),))
        

bnds_beta = (((-10, 10),)*n_cities.numpy())
bnds_rest = ((-4, 0), (-4, 0), (-1, 5), (-20, -np.log(2)-np.log(p)), (-20, -0.01), (-10, 3), (-10, -0.1))
bnds = bnds_x_0+bnds_beta+bnds_rest
string1 = ["################################", "\n"]
string2 = ["Start sequential optimization", "\n"]
string3 = ["################################", "\n"]
f= open("PAL_cineca/Data/Check/"+simulation_name+".txt", "a")
f.writelines(string1)
f.writelines(string2)
f.writelines(string3)
f.close()

seq_x_rest_to_start = x_to_start

def seq_optimization_func(seq_x_rest_to_optim):
        
    seq_x_rest_to_start[3*n_cities:] = seq_x_rest_to_optim
    
    return optimization_func(seq_x_rest_to_start)
    
seq_x_rest_to_optim = seq_x_rest_to_start[3*n_cities:]
seq_bnds = bnds[3*n_cities:]

start = time.time()
res = minimize(seq_optimization_func, seq_x_rest_to_optim, bounds = seq_bnds, method='SLSQP', options ={"eps":0.5, "maxiter":500})
print("Running time for the optimization ", str(time.time()-start))

string1 = ["################################", "\n"]
string2 = ["The max is ", "\n"]
string3 = [" "+str(res.x), "\n"] 
f= open("PAL_cineca/Data/Check/"+simulation_name+".txt", "a")
f.writelines(string1)
f.writelines(string2)
f.writelines(string3)
f.close()

seq_x_rest_to_start[3*n_cities:] = res.x

np.save("PAL_cineca/Data/Output/"+simulation_name+".npy", seq_x_rest_to_start)

