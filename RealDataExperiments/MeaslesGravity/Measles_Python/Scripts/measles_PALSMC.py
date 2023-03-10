import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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

        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[t_intermediate] + ((1-c)/(26*intermediate_steps-1))*UKbirths_t*(1-is_start_school_year_array_t_obs[t_intermediate])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        surv_prob = 1-tf.expand_dims(delta_year/(26*intermediate_steps), axis = 0)

        Lambda_t = tf.einsum("pnm,pnmk->pnmk", bar_lambda_tm1*surv_prob, K_tprev)

        bar_lambda_t = tf.reduce_sum(Lambda_t, axis =2) + alpha_t

        return bar_lambda_t, Lambda_t

    Lambda_tprev = tf.zeros((n_particles, n_cities, 4, 4))

    lambda_, Lambda_ = tf.scan(body, tf.range(0, intermediate_steps, dtype=tf.int64), initializer = (bar_lambda_tprev, Lambda_tprev)) 

    output = lambda_, Lambda_

    return output

@tf.function(jit_compile=True)
def PAL_body_run(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, n_particles, delta_year, v):

    infection_rate = PAL_compute_infection_rate(bar_lambda_tm1, is_school_term_array_t[0], pop_t, p, a, beta_bar, v, xi_t)
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

    bar_Lambda_t = (1-Q_t)*Lambda_[-1,...] + Y_t*Lambda_[-1,...]*Q_t/M_corr

    likelihood_t_tm1 = tf.reduce_sum(tfp.distributions.Poisson(M[...,2,3]).log_prob(tf.transpose(y_t)), axis = -1)

    return tf.reduce_sum(bar_Lambda_t, axis = 2), likelihood_t_tm1[0,:], bar_Lambda_t, M


def PAL_run(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
    v = (g*measles_distance_matrix)

    def body(input, t_obs):

        bar_lambda_tm1, _, _, _ = input

        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_particles, n_cities, 1))
        q_t  = Q.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,:]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs:t_obs+1]

        bar_lambda_t, likelihood_t_tm1, bar_Lambda_t, M = PAL_body_run(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, n_particles, delta_year, v)

        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[-1] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[-1])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis =2) + alpha_t

        return bar_lambda_t, likelihood_t_tm1, bar_Lambda_t, M

    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis =1)*pi_0, axis =0)*tf.ones((n_particles, n_cities, 4))
    bar_Lambda_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)
    M_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)
    likelihood_0 = tf.zeros((n_particles), dtype=tf.float32)

    bar_lambda, likelihood, bar_Lambda, M = tf.scan(body, tf.range(0, T, dtype=tf.int64), initializer = (bar_lambda_0, likelihood_0, bar_Lambda_0, M_0)) 

    return bar_lambda, likelihood, bar_Lambda, M

@tf.function(jit_compile=True)
def PAL_body_run_res(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, Q, c, n_cities, n_particles, delta_year, v):

    infection_rate = PAL_compute_infection_rate(bar_lambda_tm1, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t)
    K_t = PAL_assemble_K(h, infection_rate, rho, gamma)

    lambda_, Lambda_ = PAL_scan_intermediate(bar_lambda_tm1, K_t, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, n_particles, delta_year)

    f_xi = tf.reduce_sum(Lambda_,axis=0)[:,:,2,3]

    b = (-Q.parameters["scale"]*Q.parameters["scale"]*f_xi + Q.parameters["loc"])

    mu_r    = (b + tf.math.sqrt(b*b + 4*Q.parameters["scale"]*Q.parameters["scale"]*tf.transpose(UKmeasles_t)))/2
    mu_r_norm    = mu_r+tf.cast((mu_r==0), dtype = tf.float32)
    sigma_r = tf.math.sqrt(1/((tf.transpose(UKmeasles_t)/(mu_r_norm*mu_r_norm))+1/(Q.parameters["scale"]*Q.parameters["scale"])))

    q_t = tfp.distributions.TruncatedNormal( mu_r, sigma_r, 0, 1).sample() #Q.sample(((n_particles, n_cities))) #
    # q_t = mu_r

    q_t = tf.expand_dims(tf.expand_dims(q_t, axis =-1), axis =-1)
    Q_t_r3 = tf.concat((tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), q_t), axis =-1)
    Q_t = tf.concat((tf.zeros((tf.shape(Q_t_r3))), tf.zeros((tf.shape(Q_t_r3))), Q_t_r3, tf.zeros((tf.shape(Q_t_r3)))), axis = -2)

    y_t = tf.expand_dims(UKmeasles_t, axis =-1)
    Y_t_r3 = tf.concat((tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), y_t), axis =-1)
    Y_t = tf.concat((tf.zeros((tf.shape(Y_t_r3))), tf.zeros((tf.shape(Y_t_r3))), Y_t_r3, tf.zeros((tf.shape(Y_t_r3)))), axis = -2)

    M = tf.reduce_sum(tf.expand_dims(Q_t, axis = 0)*Lambda_, axis = 0)
    M_corr = M + tf.cast(M==0, dtype=tf.float32)

    bar_Lambda_t = (1-Q_t)*Lambda_[-1,...] + tf.where((Y_t*Lambda_[-1,...]*Q_t)==0, Y_t*Lambda_[-1,...]*Q_t, Y_t*Lambda_[-1,...]*Q_t/M)

    likelihood_t_tm1 = tfp.distributions.Poisson(M[...,2,3]).log_prob(tf.transpose(UKmeasles_t))+ Q.log_prob(q_t[...,0,0]) - tfp.distributions.TruncatedNormal( mu_r, sigma_r, 0, 1).log_prob(q_t[...,0,0])

    return tf.reduce_sum(bar_Lambda_t, axis = 2), likelihood_t_tm1, bar_Lambda_t, M, q_t


@tf.function(jit_compile=True)
def PAL_body_run_res_low(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, Q, c, n_cities, n_particles, delta_year, v):

    infection_rate = PAL_compute_infection_rate(bar_lambda_tm1, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t)
    K_t = PAL_assemble_K(h, infection_rate, rho, gamma)

    _, Lambda_ = PAL_scan_intermediate(bar_lambda_tm1, K_t, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, n_particles, delta_year)

    f_xi = tf.reduce_sum(Lambda_,axis=0)[:,:,2,3]

    b = (-Q.parameters["scale"]*Q.parameters["scale"]*f_xi + Q.parameters["loc"])

    mu_r    = (b + tf.math.sqrt(b*b + 4*Q.parameters["scale"]*Q.parameters["scale"]*tf.transpose(UKmeasles_t)))/2
    mu_r_norm    = mu_r+tf.cast((mu_r==0), dtype = tf.float32)
    sigma_r = tf.math.sqrt(1/((tf.transpose(UKmeasles_t)/(mu_r_norm*mu_r_norm))+1/(Q.parameters["scale"]*Q.parameters["scale"])))

    q_t = tfp.distributions.TruncatedNormal( mu_r, sigma_r, 0, 1).sample() #Q.sample(((n_particles, n_cities))) #
    # q_t = mu_r

    q_t = tf.expand_dims(tf.expand_dims(q_t, axis =-1), axis =-1)
    Q_t_r3 = tf.concat((tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), q_t), axis =-1)
    Q_t = tf.concat((tf.zeros((tf.shape(Q_t_r3))), tf.zeros((tf.shape(Q_t_r3))), Q_t_r3, tf.zeros((tf.shape(Q_t_r3)))), axis = -2)

    y_t = tf.expand_dims(UKmeasles_t, axis =-1)
    Y_t_r3 = tf.concat((tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), y_t), axis =-1)
    Y_t = tf.concat((tf.zeros((tf.shape(Y_t_r3))), tf.zeros((tf.shape(Y_t_r3))), Y_t_r3, tf.zeros((tf.shape(Y_t_r3)))), axis = -2)

    M = tf.reduce_sum(tf.expand_dims(Q_t, axis = 0)*Lambda_, axis = 0)

    bar_Lambda_t = (1-Q_t)*Lambda_[-1,...] + tf.where((Y_t*Lambda_[-1,...]*Q_t)==0, Y_t*Lambda_[-1,...]*Q_t, Y_t*Lambda_[-1,...]*Q_t/M)

    likelihood_t_tm1 = tfp.distributions.Poisson(M[...,2,3]).log_prob(tf.transpose(UKmeasles_t))+ Q.log_prob(q_t[...,0,0]) - tfp.distributions.TruncatedNormal( mu_r, sigma_r, 0, 1).log_prob(q_t[...,0,0])

    return likelihood_t_tm1, bar_Lambda_t


def PAL_run_likelihood_lookahead(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
    v = (g*measles_distance_matrix)
    def cond(t_obs, input):
    
        return t_obs<T

    def body(t_obs, input):

        bar_lambda_tm1, log_weights_tm1, log_alpha_tm1, loglikelihood = input 

        # Resampling with correction
        log_alpha_tm1_corrected = tf.where(tf.math.is_nan(log_alpha_tm1), -500*tf.ones(tf.shape(log_alpha_tm1)), log_alpha_tm1)
        alpha_tm1_unorm = tf.exp((log_alpha_tm1_corrected)-tf.reduce_max((log_alpha_tm1_corrected), axis =0, keepdims=True))
        alpha_tm1 = alpha_tm1_unorm/tf.reduce_sum(alpha_tm1_unorm, axis =0)

        indeces = tfp.distributions.Categorical(probs=tf.transpose(alpha_tm1)).sample(n_particles)
        res_bar_lambda_tm1 = tf.transpose(tf.gather(tf.transpose(bar_lambda_tm1, [1, 0, 2   ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2   ])
        res_log_weights_tm1 = tf.transpose(tf.gather(tf.transpose(log_weights_tm1, [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
        res_log_alpha_tm1 = tf.transpose(tf.gather(tf.transpose(log_alpha_tm1, [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
        res_log_weights_tm1 = res_log_weights_tm1 - res_log_alpha_tm1

        # t
        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs:t_obs+1]

        log_likelihood_t_tm1, bar_Lambda_t = PAL_body_run_res_low(res_bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, Q, c, n_cities, n_particles, delta_year, v)

        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[-1] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[-1])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis =2) + alpha_t

        # t+1
        t_obs = t_obs+1
        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_tp1 = Xi.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs:t_obs+1]

        log_alpha_t, _ = PAL_body_run_res_low(bar_lambda_t, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_tp1, Q, c, n_cities, n_particles, delta_year, v)

        weights_flow = tf.math.exp(res_log_weights_tm1- tf.reduce_max(res_log_weights_tm1, axis =0, keepdims=True))
        log_weights_tm1 = tf.math.log(weights_flow/tf.reduce_sum(weights_flow, axis =0, keepdims=True))

        log_weights_t = log_weights_tm1 + log_likelihood_t_tm1
        log_alpha_t = log_alpha_t + log_weights_t

        likelihood_t_tm1_norm = tf.exp((log_weights_t)-tf.reduce_max((log_weights_t), axis =0, keepdims=True))
        log_increment =  tf.reduce_sum(tf.math.log(tf.reduce_sum(likelihood_t_tm1_norm, axis =0)) + tf.reduce_max((log_weights_t), axis =0)) 

        return t_obs, (bar_lambda_t, log_weights_t, log_alpha_t, loglikelihood + log_increment)

    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis =1)*pi_0, axis =0)*tf.ones((n_particles, n_cities, 4))
    log_weights_0 = tf.zeros((n_particles, n_cities), dtype=tf.float32)
    log_alpha_0   = log_weights_0
    loglikelihood_0 = tf.zeros((1), dtype=tf.float32)

    time, output = tf.while_loop( cond, body, loop_vars=[0, (bar_lambda_0, log_weights_0, log_alpha_0, loglikelihood_0)])

    return output[3]



def PAL_run_likelihood_res(T, intermediate_steps, UKmeasles, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year):
    
    v = (g*measles_distance_matrix)

    def cond(t_obs, input):
    
        return t_obs<T

    def body(t_obs, input):

        bar_lambda_tm1, _, log_likelihood, _ = input

        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        is_school_term_array_t = is_school_term_array[t_obs,0]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:,t_obs:t_obs+1]

        bar_lambda_t, loglikelihood_t_tm1, bar_Lambda_t, M, _ = PAL_body_run_res(bar_lambda_tm1, intermediate_steps, UKmeasles_t, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, Q, c, n_cities, n_particles, delta_year, v)

        alpha_t = c*UKbirths_t*is_start_school_year_array_t_obs[-1] + ((1-c)/(26*intermediate_steps))*UKbirths_t*(1-is_start_school_year_array_t_obs[-1])
        alpha_t = tf.expand_dims(alpha_t, axis = 0)
        alpha_t = tf.concat((alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))), axis = -1)

        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis =2) + alpha_t

        loglikelihood_t_tm1 = tf.where(tf.math.is_nan(loglikelihood_t_tm1), -500*tf.ones(tf.shape(loglikelihood_t_tm1)), loglikelihood_t_tm1)
        likelihood_t_tm1_flow = tf.exp((loglikelihood_t_tm1)-tf.reduce_max((loglikelihood_t_tm1), axis =0, keepdims=True))
        norm_weights = likelihood_t_tm1_flow/tf.reduce_sum(likelihood_t_tm1_flow, axis =0)

        indeces = tfp.distributions.Categorical(probs=tf.transpose(norm_weights)).sample(n_particles)
        res_bar_lambda_t = tf.transpose(tf.gather(tf.transpose(bar_lambda_t, [1, 0, 2   ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2   ])
        res_bar_Lambda_t = tf.transpose(tf.gather(tf.transpose(bar_Lambda_t, [1, 0, 2, 3]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2, 3])

        likelihood_t_tm1_norm = tf.exp((loglikelihood_t_tm1)-tf.reduce_max((loglikelihood_t_tm1), axis =0, keepdims=True))
        log_increment =  tf.reduce_sum(tf.math.log(tf.reduce_mean(likelihood_t_tm1_norm, axis =0)) + tf.reduce_max((loglikelihood_t_tm1), axis =0)) 
        
        return t_obs+1, (res_bar_lambda_t, res_bar_Lambda_t, log_likelihood + log_increment, M)

    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis =1)*pi_0, axis =0)*tf.ones((n_particles, n_cities, 4))
    bar_Lambda_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)
    likelihood_0 = tf.zeros((1), dtype=tf.float32)
    M_0 = tf.zeros((n_particles, n_cities, 4, 4), dtype=tf.float32)

    time, output = tf.while_loop( cond, body, loop_vars=[0, (bar_lambda_0, bar_Lambda_0, likelihood_0, M_0)])

    return output[2]
