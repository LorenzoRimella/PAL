import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd

import matplotlib.pyplot as plt


def log_factorial(y_t):
    
    def cond(city_index, log_sum):

        return city_index<40

    def body(city_index, log_sum):

        return city_index + 1, log_sum + tf.reduce_sum(tf.math.log(tf.linspace(tf.constant(1, dtype = tf.float32), y_t[city_index,0,0], tf.cast(y_t[city_index,0,0], dtype = tf.int64))))

    output = tf.while_loop( cond, body, loop_vars=[0, tf.constant(0, dtype = tf.float32)])

    return output[1]

def log_correction(T, UKmeasles):

    def body(input, t_obs):

        UKmeasles_t = UKmeasles[:,t_obs+1:t_obs+2]

        return log_factorial(tf.expand_dims(UKmeasles_t, axis =-1))

    return tf.scan(body, tf.range(0, T, dtype=tf.int64), initializer = (tf.constant(0, dtype = tf.float32))) 

@tf.function(jit_compile=True)
def PAL_compute_infection_rate(bar_lambda_tm1, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t):

    infected_prop_t = tf.einsum("pcm,c->pcm", bar_lambda_tm1[...,2:3], 1/pop_t)

    beta_t = (1+2*(1-p)*a)*beta_bar*is_school_term_array_t + (1-2*p*a)*beta_bar*(1-is_school_term_array_t)

    spatial_infection = infected_prop_t + tf.reduce_sum((v/pop_t)*(tf.transpose(infected_prop_t, perm = [0, 2, 1]) - infected_prop_t), axis = 2, keepdims= True)

    infection_rate = beta_t*xi_t*spatial_infection

    return infection_rate

@tf.function(jit_compile=True)
def PAL_assemble_K(h, infection_rate, rho, gamma):
    
    prob_inf = tf.expand_dims(1-tf.exp(-h*infection_rate), axis = 2)
    K_r1 = tf.concat((1-prob_inf, prob_inf, tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf))), axis = -1)

    prob_latent = tf.ones((tf.shape(prob_inf)))*tf.expand_dims(tf.expand_dims(1-tf.exp(-h*rho), axis = 2), axis =0 )
    K_r2 = tf.concat((tf.zeros(tf.shape(prob_inf)), 1-prob_latent, prob_latent, tf.zeros(tf.shape(prob_inf))), axis = -1)

    prob_recover = tf.ones((tf.shape(prob_inf)))*tf.expand_dims(tf.expand_dims(1-tf.exp(-h*gamma), axis = 2), axis =0 )
    K_r3 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), 1-prob_recover, prob_recover, ), axis = -1)

    K_r4 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), tf.ones(tf.shape(prob_inf))), axis = -1)

    K_t = tf.concat((K_r1, K_r2, K_r3, K_r4), axis = 2)

    return K_t

@tf.function(jit_compile=True)
def PAL_scan_intermediate(bar_lambda_tprev, K_tprev, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, n_particles, delta_year):

    def body(input, t_intermediate):

        bar_lambda_tm1, Lambda_tm1 = input

        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[t_intermediate] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[t_intermediate])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        surv_prob = 1-tf.expand_dims(delta_year/(26*intermediate_steps), axis = 0)

        Lambda_t = tf.einsum("pnm,pnmk->pnmk", bar_lambda_tm1*surv_prob + alpha_t, K_tprev)

        bar_lambda_t = tf.reduce_sum(Lambda_t, axis =2) #+ alpha_t

        return bar_lambda_t, Lambda_t

    Lambda_tprev = tf.zeros((n_particles, n_cities, 4, 4))

    lambda_, Lambda_ = tf.scan(body, tf.range(0, intermediate_steps, dtype=tf.int64), initializer = (bar_lambda_tprev, Lambda_tprev)) 

    output = lambda_, Lambda_

    return output

@tf.function(jit_compile=True)
def PAL_body_run(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, n_particles, delta_year, v):

    infection_rate = PAL_compute_infection_rate(bar_lambda_tm1, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t)
    K_t = PAL_assemble_K(h, infection_rate, rho, gamma)

    lambda_, Lambda_ = PAL_scan_intermediate(bar_lambda_tm1, K_t, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, n_particles, delta_year)

    q_t = tf.expand_dims(q_t, axis =-1)
    Q_t_r3 = tf.concat((tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), q_t), axis =-1)
    Q_t = tf.concat((tf.zeros((tf.shape(Q_t_r3))), tf.zeros((tf.shape(Q_t_r3))), Q_t_r3, tf.zeros((tf.shape(Q_t_r3)))), axis = -2)

    y_t = tf.expand_dims(UKmeasles_t, axis =-1)
    Y_t_r3 = tf.concat((tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), y_t), axis =-1)
    Y_t = tf.concat((tf.zeros((tf.shape(Y_t_r3))), tf.zeros((tf.shape(Y_t_r3))), Y_t_r3, tf.zeros((tf.shape(Y_t_r3)))), axis = -2)

    M = tf.reduce_sum(tf.expand_dims(Q_t, axis = 0)*Lambda_, axis = 0)
    M_corr = M + tf.cast(M==0, dtype=tf.float32)

    bar_Lambda_t = (1-Q_t)*Lambda_[-1,...] + tf.where((Y_t*Lambda_[-1,...]*Q_t)==0, Y_t*Lambda_[-1,...]*Q_t, Y_t*Lambda_[-1,...]*Q_t/M)

    likelihood_t_tm1_no_corr = -tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(M, axis = -1), axis = -1), axis = -1) + tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.math.log(M_corr)*Y_t, axis = -1), axis = -1), axis = -1) #- log_factorial(y_t)

    return tf.reduce_sum(bar_Lambda_t, axis = 2), likelihood_t_tm1_no_corr, bar_Lambda_t


def PAL_run(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, log_factorial_vec, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
    v = (g*measles_distance_matrix)

    def body(input, t_obs):

        bar_lambda_tm1, _, _ = input

        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_particles, n_cities, 1))
        q_t  = Q.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs+1:t_obs+2]

        bar_lambda_t, likelihood_t_tm1_no_corr, bar_Lambda_t = PAL_body_run(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, n_particles, delta_year, v)

        birth_index = tf.cast((t_obs+1)/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]
        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[-1] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[-1])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis =2) #+ alpha_t

        return bar_lambda_t, likelihood_t_tm1_no_corr- log_factorial_vec[t_obs], bar_Lambda_t

    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis =1)*pi_0, axis =0)*tf.ones((n_particles, n_cities, 4))
    bar_Lambda_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)
    likelihood_0 = tf.zeros((n_particles), dtype=tf.float32)

    bar_lambda, likelihood, bar_Lambda = tf.scan(body, tf.range(0, T, dtype=tf.int64), initializer = (bar_lambda_0, likelihood_0, bar_Lambda_0)) 

    return bar_lambda, likelihood, bar_Lambda


def PAL_run_likelihood(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, log_factorial_vec, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
    v = (g*measles_distance_matrix)

    def cond(t_obs, input):
    
        return t_obs<T

    def body(t_obs, input):

        bar_lambda_tm1, _, log_likelihood = input

        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_particles, n_cities, 1))
        q_t  = Q.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs+1:t_obs+2]

        bar_lambda_t, likelihood_t_tm1_no_corr, bar_Lambda_t = PAL_body_run(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, n_particles, delta_year, v)

        birth_index = tf.cast((t_obs+1)/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]
        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[-1] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[-1])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis =2) + alpha_t
        
        return t_obs+1, (bar_lambda_t, bar_Lambda_t, log_likelihood + likelihood_t_tm1_no_corr- log_factorial_vec[t_obs])

    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis =1)*pi_0, axis =0)*tf.ones((n_particles, n_cities, 4))
    bar_Lambda_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)
    likelihood_0 = tf.zeros((n_particles), dtype=tf.float32)

    time, output = tf.while_loop( cond, body, loop_vars=[0, (bar_lambda_0, bar_Lambda_0, likelihood_0)])

    return output[2]

@tf.function(jit_compile=True)
def PAL_body_run_res(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, Q, c, n_cities, n_particles, delta_year, v):

    infection_rate = PAL_compute_infection_rate(bar_lambda_tm1, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t)
    K_t = PAL_assemble_K(h, infection_rate, rho, gamma)

    lambda_, Lambda_ = PAL_scan_intermediate(bar_lambda_tm1, K_t, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, n_particles, delta_year)

    f_xi = tf.reduce_sum(Lambda_,axis=0)[:,:,2,3]

    b = (-Q.variance()*f_xi + Q.mean())

    mu_r    = (b + tf.math.sqrt(b*b + 4*Q.variance()*tf.transpose(UKmeasles_t)))/2
    mu_r    = mu_r+tf.cast((mu_r==0), dtype = tf.float32)
    sigma_r = tf.math.sqrt(1/((tf.transpose(UKmeasles_t)/(mu_r*mu_r))+1/Q.variance()))

    q_t = tfp.distributions.TruncatedNormal( mu_r, sigma_r, 0, 1).sample()

    q_t = tf.expand_dims(tf.expand_dims(q_t, axis =-1), axis =-1)
    Q_t_r3 = tf.concat((tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), q_t), axis =-1)
    Q_t = tf.concat((tf.zeros((tf.shape(Q_t_r3))), tf.zeros((tf.shape(Q_t_r3))), Q_t_r3, tf.zeros((tf.shape(Q_t_r3)))), axis = -2)

    y_t = tf.expand_dims(UKmeasles_t, axis =-1)
    Y_t_r3 = tf.concat((tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), y_t), axis =-1)
    Y_t = tf.concat((tf.zeros((tf.shape(Y_t_r3))), tf.zeros((tf.shape(Y_t_r3))), Y_t_r3, tf.zeros((tf.shape(Y_t_r3)))), axis = -2)

    M = tf.reduce_sum(tf.expand_dims(Q_t, axis = 0)*Lambda_, axis = 0)
    M_corr = M + tf.cast(M==0, dtype=tf.float32)

    bar_Lambda_t = (1-Q_t)*Lambda_[-1,...] + tf.where((Y_t*Lambda_[-1,...]*Q_t)==0, Y_t*Lambda_[-1,...]*Q_t, Y_t*Lambda_[-1,...]*Q_t/M)

    likelihood_t_tm1_no_corr = -tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(M, axis = -1), axis = -1), axis = -1) + tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.math.log(M_corr)*Y_t, axis = -1), axis = -1), axis = -1) #- log_factorial(y_t)

    return tf.reduce_sum(bar_Lambda_t, axis = 2), likelihood_t_tm1_no_corr, bar_Lambda_t


def PAL_run_res(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, log_factorial_vec, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
    v = (g*measles_distance_matrix)

    def body(input, t_obs):

        bar_lambda_tm1, _, _ = input

        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs+1:t_obs+2]

        bar_lambda_t, likelihood_t_tm1_no_corr, bar_Lambda_t = PAL_body_run_res(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, Q, c, n_cities, n_particles, delta_year, v)

        birth_index = tf.cast((t_obs+1)/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]
        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[-1] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[-1])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis =2) + alpha_t

        likelihood_t_tm1_no_corr_flow = tf.exp((likelihood_t_tm1_no_corr)-tf.reduce_max((likelihood_t_tm1_no_corr)))
        norm_weights = likelihood_t_tm1_no_corr_flow/tf.reduce_sum(likelihood_t_tm1_no_corr_flow)
        indeces = tfp.distributions.Categorical(probs=norm_weights).sample(n_particles)

        bar_lambda_t = tf.gather(bar_lambda_t, indeces, axis = 0)
        likelihood_t_tm1_no_corr = tf.gather(likelihood_t_tm1_no_corr, indeces, axis = 0)
        bar_Lambda_t = tf.gather(bar_Lambda_t, indeces, axis = 0)

        return bar_lambda_t, likelihood_t_tm1_no_corr- log_factorial_vec[t_obs], bar_Lambda_t

    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis =1)*pi_0, axis =0)*tf.ones((n_particles, n_cities, 4))
    bar_Lambda_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)
    likelihood_0 = tf.zeros((n_particles), dtype=tf.float32)

    bar_lambda, likelihood, bar_Lambda = tf.scan(body, tf.range(0, T, dtype=tf.int64), initializer = (bar_lambda_0, likelihood_0, bar_Lambda_0)) 

    return bar_lambda, likelihood, bar_Lambda


def PAL_run_likelihood_res(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, log_factorial_vec, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
    v = (g*measles_distance_matrix)

    def cond(t_obs, input):
    
        return t_obs<T

    def body(t_obs, input):

        bar_lambda_tm1, _, log_likelihood = input

        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs+1:t_obs+2]

        bar_lambda_t, likelihood_t_tm1_no_corr, bar_Lambda_t = PAL_body_run_res(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, Q, c, n_cities, n_particles, delta_year, v)

        birth_index = tf.cast((t_obs+1)/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]
        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[-1] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[-1])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis =2) + alpha_t

        likelihood_t_tm1_no_corr_flow = tf.exp((likelihood_t_tm1_no_corr)-tf.reduce_max((likelihood_t_tm1_no_corr)))
        norm_weights = likelihood_t_tm1_no_corr_flow/tf.reduce_sum(likelihood_t_tm1_no_corr_flow)
        indeces = tfp.distributions.Categorical(probs=norm_weights).sample(n_particles)

        bar_lambda_t = tf.gather(bar_lambda_t, indeces, axis = 0)
        likelihood_t_tm1_no_corr = tf.gather(likelihood_t_tm1_no_corr, indeces, axis = 0)
        bar_Lambda_t = tf.gather(bar_Lambda_t, indeces, axis = 0)
        
        return t_obs+1, (bar_lambda_t, bar_Lambda_t, log_likelihood + likelihood_t_tm1_no_corr- log_factorial_vec[t_obs])

    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis =1)*pi_0, axis =0)*tf.ones((n_particles, n_cities, 4))
    bar_Lambda_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)
    likelihood_0 = tf.zeros((n_particles), dtype=tf.float32)

    time, output = tf.while_loop( cond, body, loop_vars=[0, (bar_lambda_0, bar_Lambda_0, likelihood_0)])

    return output[2]