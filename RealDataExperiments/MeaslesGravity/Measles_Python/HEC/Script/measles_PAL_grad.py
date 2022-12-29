import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd

import matplotlib.pyplot as plt


def PAL_compute_infection_rate_grad(bar_lambda_tm1, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t):

    infected_prop_t = tf.einsum("pcm,c->pcm", bar_lambda_tm1[...,2:3], 1/pop_t)

    beta_t = (1+2*(1-p)*a)*beta_bar*is_school_term_array_t + (1-2*p*a)*beta_bar*(1-is_school_term_array_t)

    spatial_infection = infected_prop_t + tf.reduce_sum((v/pop_t)*(tf.transpose(infected_prop_t, perm = [0, 2, 1]) - infected_prop_t), axis = 2, keepdims= True)

    infection_rate = beta_t*xi_t*spatial_infection

    return infection_rate


def PAL_assemble_K_grad(h, infection_rate, rho, gamma):
    
    prob_inf = tf.expand_dims(1-tf.exp(-h*infection_rate), axis = 2)
    K_r1 = tf.concat((1-prob_inf, prob_inf, tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf))), axis = -1)

    prob_latent = tf.ones((tf.shape(prob_inf)))*tf.expand_dims(tf.expand_dims(1-tf.exp(-h*rho), axis = 2), axis =0 )
    K_r2 = tf.concat((tf.zeros(tf.shape(prob_inf)), 1-prob_latent, prob_latent, tf.zeros(tf.shape(prob_inf))), axis = -1)

    prob_recover = tf.ones((tf.shape(prob_inf)))*tf.expand_dims(tf.expand_dims(1-tf.exp(-h*gamma), axis = 2), axis =0 )
    K_r3 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), 1-prob_recover, prob_recover, ), axis = -1)

    K_r4 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), tf.ones(tf.shape(prob_inf))), axis = -1)

    K_t = tf.concat((K_r1, K_r2, K_r3, K_r4), axis = 2)

    return K_t


def PAL_scan_intermediate_grad(bar_lambda_tprev, K_tprev, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, n_particles, delta_year):

    def body(input, t_intermediate):

        bar_lambda_tm1, Lambda_tm1 = input

        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[t_intermediate] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[t_intermediate])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        surv_prob = 1-tf.expand_dims(delta_year/(26*intermediate_steps), axis = 0)

        Lambda_t = tf.einsum("pnm,pnmk->pnmk", bar_lambda_tm1*surv_prob, K_tprev)

        bar_lambda_t = tf.reduce_sum(Lambda_t, axis =2) + alpha_t

        return bar_lambda_t, Lambda_t

    Z_tprev = tf.zeros((n_particles, n_cities, 4, 4))

    lambda_, Z_ = tf.scan(body, tf.range(0, intermediate_steps, dtype=tf.int64), initializer = (bar_lambda_tprev, Z_tprev)) 

    output = lambda_, Z_

    return output


def PAL_body_run_grad(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, n_particles, delta_year, v):

    infection_rate = PAL_compute_infection_rate_grad(bar_lambda_tm1, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t)
    K_t = PAL_assemble_K_grad(h, infection_rate, rho, gamma)

    lambda_, Lambda_ = PAL_scan_intermediate_grad(bar_lambda_tm1, K_t, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, n_particles, delta_year)

    q_t = tf.expand_dims(q_t, axis =-1)
    Q_t_r3 = tf.concat((tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), q_t), axis =-1)
    Q_t = tf.concat((tf.zeros((tf.shape(Q_t_r3))), tf.zeros((tf.shape(Q_t_r3))), Q_t_r3, tf.zeros((tf.shape(Q_t_r3)))), axis = -2)

    y_t = tf.expand_dims(UKmeasles_t, axis =-1)
    Y_t_r3 = tf.concat((tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), y_t), axis =-1)
    Y_t = tf.concat((tf.zeros((tf.shape(Y_t_r3))), tf.zeros((tf.shape(Y_t_r3))), Y_t_r3, tf.zeros((tf.shape(Y_t_r3)))), axis = -2)

    M = tf.reduce_sum(tf.expand_dims(Q_t, axis = 0)*Lambda_, axis = 0)
    M_corr = M + tf.cast(M==0, dtype=tf.float32)

    bar_Lambda_t = (1-Q_t)*Lambda_[-1,...] + Y_t*Lambda_[-1,...]*Q_t/M_corr

    likelihood_t_tm1_no_corr = -tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(M, axis = -1), axis = -1), axis = -1) + tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.math.log(M_corr)*Y_t, axis = -1), axis = -1), axis = -1)

    return tf.reduce_sum(bar_Lambda_t, axis = 2), likelihood_t_tm1_no_corr, bar_Lambda_t


def PAL_run_likelihood_grad(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, log_factorial_vec, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
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

        bar_lambda_t, likelihood_t_tm1_no_corr, bar_Lambda_t = PAL_body_run_grad(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, n_particles, delta_year, v)

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