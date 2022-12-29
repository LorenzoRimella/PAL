import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import sys
sys.path.append('Measles/Scripts/')
from measles_code import *
from measles_PAL import *
from measles_PAL_grad import *

#############################################
# Simulate the data
#############################################
UKbirths_array = np.load("Measles/Data/Input/UKbirths_array.npy")
UKpop_array = np.load("Measles/Data/Input/UKpop_array.npy")
measles_distance_matrix_array = np.load("Measles/Data/Input/measles_distance_matrix_array.npy")

UKbirths = tf.convert_to_tensor(UKbirths_array, dtype = tf.float32)
UKpop = tf.convert_to_tensor(UKpop_array, dtype = tf.float32)
measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array, dtype = tf.float32)

term   = tf.convert_to_tensor([6, 99, 115, 198, 252, 299, 308, 355, 366], dtype = tf.float32)
school = tf.convert_to_tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype = tf.float32)

n_cities = tf.constant(40, dtype = tf.int64)

pi_0_1 = 0.01
pi_0_2 = 0.00005
pi_0_3 = 0.00004
pi_0 = tf.convert_to_tensor([[pi_0_1, pi_0_2, pi_0_3, 1 - pi_0_1 - pi_0_2 - pi_0_3]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)

initial_pop = UKpop[:,0]

beta_bar  = tf.convert_to_tensor(np.random.normal(10, 0.5, (n_cities,1)), dtype = tf.float32)
rho   = tf.convert_to_tensor([[1/7]], dtype = tf.float32)*tf.ones((n_cities, 1), dtype = tf.float32)
gamma = tf.convert_to_tensor([[1/7]], dtype = tf.float32)*tf.ones((n_cities, 1), dtype = tf.float32)

g = tf.convert_to_tensor(np.random.normal(150, 10, (n_cities,1)), dtype = tf.float32)

p = tf.constant(0.759, dtype = tf.float32)
a = tf.constant(0.3,   dtype = tf.float32)
c = tf.constant(0.4,   dtype = tf.float32)

Xi = tfp.distributions.Gamma(concentration = 50, rate = 50)
Q  = tfp.distributions.TruncatedNormal( 0.5, 0.1, 0, 1)

delta_year = tf.convert_to_tensor([[1/50]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)

intermediate_steps = 4
T = 600

is_school_term_array, is_start_school_year_array = school_term_and_school_year(T, intermediate_steps, term, school)

is_school_term_array = tf.convert_to_tensor(is_school_term_array, dtype = tf.float32)
is_start_school_year_array = tf.convert_to_tensor(is_start_school_year_array, dtype = tf.float32)

intermediate_steps = tf.constant(4, dtype = tf.float32)
h = tf.constant(14/tf.cast(intermediate_steps, dtype = tf.float32), dtype = tf.float32)
T = tf.constant(540, dtype = tf.float32)

X_t, Y_t = run(T, intermediate_steps, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, h, rho, gamma, Xi, Q, c, n_cities, delta_year)

np.save("Measles/Data/Output/synthetic_data.npy", Y_t)
np.save("Measles/Data/Output/param_beta_bar.npy", beta_bar.numpy())
np.save("Measles/Data/Output/param_g.npy", g.numpy())

synthetic_data = tf.transpose(Y_t[...,0])

T=500

log_factorial_vec =  log_correction(T, synthetic_data)


#########################################################
# Gradient
#########################################################

n_cities = tf.constant(40, dtype = tf.int64)

pi_0_1_init = 0.01
pi_0_2_init = 0.00005
pi_0_3_init = 0.00004
pi_0_init = tf.convert_to_tensor([[pi_0_1_init, pi_0_2_init, pi_0_3_init, 1 - pi_0_1_init - pi_0_2_init - pi_0_3_init]], dtype = tf.float32)
log_pi_0_init = tf.Variable(tf.math.log(pi_0_init))

initial_pop = UKpop[:,0]

beta_bar_init  = tf.convert_to_tensor(np.random.normal(7, 0.5, (n_cities,1)), dtype = tf.float32)
log_beta_bar_init  = tf.Variable(tf.math.log(beta_bar_init))

rho_init       = tf.convert_to_tensor([[1/5]], dtype = tf.float32)
log_rho_init       = tf.Variable(tf.math.log(rho_init))

gamma_init     = tf.convert_to_tensor([[1/10]], dtype = tf.float32)
log_gamma_init     = tf.Variable(tf.math.log(gamma_init))

g_init = tf.convert_to_tensor(np.random.normal(250, 10, (n_cities,1)), dtype = tf.float32)
log_g_init = tf.Variable(tf.math.log(g_init))

p_init = tf.constant(0.5, dtype = tf.float32)
log_p_init = tf.Variable(tf.math.log(p_init))

a_init = tf.constant(0.1,   dtype = tf.float32)
log_a_init = tf.Variable(tf.math.log(a_init))

c_init = tf.constant(0.7,   dtype = tf.float32)
log_c_init = tf.Variable(tf.math.log(c_init))

gamma_par_init = tf.constant(20,   dtype = tf.float32)
log_gamma_par_init = tf.Variable(tf.math.log(gamma_par_init))

gauss_mean_par_init = tf.constant(0.3,   dtype = tf.float32)
log_gauss_mean_par_init = tf.Variable(tf.math.log(gauss_mean_par_init))

gauss_sigma_par_init = tf.constant(0.5,   dtype = tf.float32)
log_gauss_sigma_par_init = tf.Variable(tf.math.log(gauss_sigma_par_init))

delta_year = tf.convert_to_tensor([[1/50]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)


pi_0_list =[]
pi_0_list.append(tf.math.exp(log_pi_0_init.numpy()))

beta_bar_list =[]
beta_bar_list.append(tf.math.exp(log_beta_bar_init.numpy()))

rho_list =[]
rho_list.append(tf.math.exp(log_rho_init.numpy()))

gamma_list =[]
gamma_list.append(tf.math.exp(log_gamma_init.numpy()))

g_list =[]
g_list.append(tf.math.exp(log_g_init.numpy()))

p_list =[]
p_list.append(tf.math.exp(log_p_init.numpy()))

a_list =[]
a_list.append(tf.math.exp(log_a_init.numpy()))

c_list =[]
c_list.append(tf.math.exp(log_c_init.numpy()))

gamma_par_list =[]
gamma_par_list.append(tf.math.exp(log_gamma_par_init.numpy()))

gauss_mean_par_list =[]
gauss_mean_par_list.append(tf.math.exp(log_gauss_mean_par_init.numpy()))

gauss_sigma_par_list =[]
gauss_sigma_par_list.append(tf.math.exp(log_gauss_sigma_par_init.numpy()))


n_particles = 400
T = 400

loss_list = []


for iter in range(10000):

    with tf.GradientTape() as tape:
        pi_0_init = tf.math.exp(log_pi_0_init)
        beta_bar_init = tf.math.exp(log_beta_bar_init)
        rho_init = tf.math.exp(log_rho_init)
        gamma_init = tf.math.exp(log_gamma_init)
        g_init = tf.math.exp(log_g_init)
        p_init = tf.math.exp(log_p_init)
        a_init = tf.math.exp(log_a_init)
        c_init = tf.math.exp(log_c_init)
        gamma_par_init = tf.math.exp(log_gamma_par_init)
        gauss_mean_par_init = tf.math.exp(log_gauss_mean_par_init)
        gauss_sigma_par_init = tf.math.exp(log_gauss_sigma_par_init)
        
        Xi = tfp.distributions.Gamma(concentration = gamma_par_init, rate = gamma_par_init)
        Q  = tfp.distributions.TruncatedNormal( gauss_mean_par_init, gauss_sigma_par_init, 0, 1)

        pi_0_init_transform = pi_0_init*tf.ones((n_cities, 4), dtype = tf.float32)
        rho_init_transform  = rho_init*tf.ones((n_cities, 1), dtype = tf.float32)
        gamma_init_transform = gamma_init*tf.ones((n_cities, 1), dtype = tf.float32)

        like = PAL_run_likelihood_grad(T, intermediate_steps, synthetic_data, UKbirths, UKpop, g_init, measles_distance_matrix, initial_pop, pi_0_init_transform, beta_bar_init, p_init, a_init, is_school_term_array, is_start_school_year_array, log_factorial_vec, h, rho_init_transform, gamma_init_transform, Xi, Q, c_init, n_cities, n_particles, delta_year)
        loss = -tf.reduce_mean(like)/T

    string = [str(iter)+" Loss: "+str(loss.numpy()), "\n"]
    f= open("Loss_check.txt", "a")
    f.writelines(string)
    f.close()

    loss_list.append(loss)

    grad_param_list = tape.gradient(loss, [log_pi_0_init, log_beta_bar_init, log_rho_init, log_gamma_init, log_g_init, log_p_init, log_a_init, log_c_init, log_gamma_par_init, log_gauss_mean_par_init, log_gauss_sigma_par_init])

    pi_0_init_updated = tf.math.exp(log_pi_0_init - 1e-4*grad_param_list[0])
    new_log_pi_3 = tf.math.log(1-tf.reduce_sum(pi_0_init_updated[0,:3]))
    log_pi_0_init_updated = tf.math.log(pi_0_init_updated)*tf.convert_to_tensor([[1, 1, 1, 0]], dtype = tf.float32) + new_log_pi_3*tf.convert_to_tensor([[0, 0, 0, 1]], dtype = tf.float32)
    log_pi_0_init     = tf.Variable(log_pi_0_init_updated)

    log_beta_bar_init = tf.Variable(log_beta_bar_init - grad_param_list[1]*1e-4)
    log_rho_init      = tf.Variable(log_rho_init      - grad_param_list[2]*1e-4)
    log_gamma_init    = tf.Variable(log_gamma_init    - grad_param_list[3]*1e-4)

    log_g_init = tf.Variable(log_g_init - grad_param_list[4]*1e-2)
    log_p_init = tf.Variable(log_p_init - grad_param_list[5]*1e-4)
    log_a_init = tf.Variable(log_a_init - grad_param_list[6]*1e-4)
    log_c_init = tf.Variable(log_c_init - grad_param_list[7]*1e-4)

    log_gamma_par_init       = tf.Variable(log_gamma_par_init       - grad_param_list[8]*1e-4)
    log_gauss_mean_par_init  = tf.Variable(log_gauss_mean_par_init  - grad_param_list[9]*1e-4)
    log_gauss_sigma_par_init = tf.Variable(log_gauss_sigma_par_init - grad_param_list[10]*1e-4)


    pi_0_list.append(tf.math.exp(log_pi_0_init.numpy()))
    beta_bar_list.append(tf.math.exp(log_beta_bar_init.numpy()))
    rho_list.append(tf.math.exp(log_rho_init.numpy()))
    gamma_list.append(tf.math.exp(log_gamma_init.numpy()))
    g_list.append(tf.math.exp(log_g_init.numpy()))
    p_list.append(tf.math.exp(log_p_init.numpy()))
    a_list.append(tf.math.exp(log_a_init.numpy()))
    c_list.append(tf.math.exp(log_c_init.numpy()))
    gamma_par_list.append(tf.math.exp(log_gamma_par_init.numpy()))
    gauss_mean_par_list.append(tf.math.exp(log_gauss_mean_par_init.numpy()))
    gauss_sigma_par_list.append(tf.math.exp(log_gauss_sigma_par_init.numpy()))

    if iter%100:
        np.save("Measles/Data/Output/pi_0_list.npy", pi_0_list)
        np.save("Measles/Data/Output/beta_bar_list.npy", beta_bar_list)
        np.save("Measles/Data/Output/rho_list.npy", rho_list)
        np.save("Measles/Data/Output/gamma_list.npy", gamma_list)
        np.save("Measles/Data/Output/g_list.npy", g_list)
        np.save("Measles/Data/Output/p_list.npy", p_list)
        np.save("Measles/Data/Output/a_list.npy", a_list)
        np.save("Measles/Data/Output/c_list.npy", c_list)
        np.save("Measles/Data/Output/gamma_par_list.npy", gamma_par_list)
        np.save("Measles/Data/Output/gauss_mean_par_list.npy", gauss_mean_par_list)
        np.save("Measles/Data/Output/gauss_sigma_par_list.npy", gauss_sigma_par_list)
