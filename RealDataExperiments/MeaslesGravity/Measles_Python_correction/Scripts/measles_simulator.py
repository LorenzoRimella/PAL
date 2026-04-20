import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def is_school_term(t_days, term, school):

    t_days_norm = t_days%365
    time = int(np.sum(term<t_days_norm))

    return school[time]

def is_start_school_year(t_days):

    t_days_norm = t_days%365
    
    return t_days_norm==248.5

def school_term_and_school_year(T, intermediate_steps, term, school):

    is_school_term_array = np.zeros((T, intermediate_steps))
    is_start_school_year_array = np.zeros((T, intermediate_steps))
    times_total = np.zeros((T*intermediate_steps))
    times_obs = np.zeros((T))
    index = 0

    for t_obs in range(0, T):

        for t_intermediate in range(1, intermediate_steps+1):

            t_days = 14*t_obs + t_intermediate*(14/intermediate_steps) + tf.math.floor(t_obs/26)
            times_total[index] = t_days
            index = index + 1

            is_school_term_array[t_obs, t_intermediate-1] = (is_school_term(t_days, term, school))
            is_start_school_year_array[t_obs, t_intermediate-1] = (is_start_school_year(t_days))

        times_obs[t_obs] = t_days
          
    return is_school_term_array, is_start_school_year_array, times_total, times_obs


@tf.function(jit_compile=True)
def compute_infection_rate(X_tm1_bar, is_school_term_array_t, pop_t, p, a, beta_bar, v, xi_t):

    infected_prop_t = X_tm1_bar[:,2:3]/pop_t

    beta_t = (1+2*(1-p)*a)*beta_bar*is_school_term_array_t + (1-2*p*a)*beta_bar*(1-is_school_term_array_t)

    spatial_infection = infected_prop_t + tf.reduce_sum((v/pop_t)*(tf.transpose(infected_prop_t) - infected_prop_t), axis = 1, keepdims= True)

    infection_rate = beta_t*xi_t*spatial_infection

    return infection_rate

@tf.function(jit_compile=True)
def assemble_K(h, infection_rate, rho, gamma):
    
    prob_inf = tf.expand_dims(1-tf.exp(-h*infection_rate), axis = 2)
    K_r1 = tf.concat((1-prob_inf, prob_inf, tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf))), axis = -1)

    prob_latent = tf.expand_dims(1-tf.exp(-h*rho), axis = 2)
    K_r2 = tf.concat((tf.zeros(tf.shape(prob_inf)), 1-prob_latent, prob_latent, tf.zeros(tf.shape(prob_inf))), axis = -1)

    prob_recover = tf.expand_dims(1-tf.exp(-h*gamma), axis = 2)
    K_r3 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), 1-prob_recover, prob_recover, ), axis = -1)

    K_r4 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), tf.ones(tf.shape(prob_inf))), axis = -1)

    K_t = tf.concat((K_r1, K_r2, K_r3, K_r4), axis = 1)

    return K_t

@tf.function(jit_compile=True)
def sample_multinomial(X_tm1_bar, K_t):

    return tfp.distributions.Multinomial( total_count = X_tm1_bar, probs = K_t).sample()

@tf.function(jit_compile=True)
def sample_poisson(alpha_tm1):

    return tfp.distributions.Poisson(rate = alpha_tm1).sample()

@tf.function(jit_compile=True)
def sample_binomial_death(exp_count, succ_prob):
    
    return tfp.distributions.Binomial( total_count = exp_count, probs = succ_prob).sample()

@tf.function(jit_compile=True)
def scan_intermediate(X_tprev, K_tprev, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, delta_year):

    def body(input, t_intermediate):

        X_tm1, Z_tm1 = input

        alpha_tm1 = c*UKbirths_t*is_start_school_year_array_t_obs[t_intermediate] + ((1-c)/(26*intermediate_steps-1))*UKbirths_t*(1-is_start_school_year_array_t_obs[t_intermediate])

        birth_tm1 = alpha_tm1
        X_birth_tm1 = tf.concat((birth_tm1, tf.zeros(tf.shape(birth_tm1)), tf.zeros(tf.shape(birth_tm1)), tf.zeros(tf.shape(birth_tm1))), axis = -1)

        death_prob = (delta_year/(26*intermediate_steps))
        death_tm1 = sample_binomial_death(X_tm1, death_prob)

        X_tm1_bar = X_tm1 - death_tm1

        Z_t = sample_multinomial(X_tm1_bar, K_tprev)
        X_t = tf.reduce_sum(Z_t, axis = 1) + X_birth_tm1

        return X_t, Z_t

    Z_tprev = tf.zeros((n_cities, 4, 4))

    X, Z = tf.scan(body, tf.range(0, intermediate_steps, dtype=tf.int64), initializer = (X_tprev, Z_tprev)) 

    output = X[-1,...], tf.reduce_sum(Z, axis = 0)

    return output

@tf.function(jit_compile=True)
def sample_binomial_obs(exp_count, succ_prob):
    
    return tfp.distributions.Binomial( total_count = exp_count, probs = succ_prob).sample()

@tf.function(jit_compile=True)
def body_run(X_tm1, intermediate_steps, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, delta_year, v):

    infection_rate = compute_infection_rate(X_tm1, is_school_term_array_t[0], pop_t, p, a, beta_bar, v, xi_t)
    K_t = assemble_K(h, infection_rate, rho, gamma)

    X_t, Z_t = scan_intermediate(X_tm1, K_t, is_start_school_year_array_t_obs, intermediate_steps, UKbirths_t, c, n_cities, delta_year)

    Y_t = sample_binomial_obs(Z_t[:,2,3:], q_t) 

    return X_t, Y_t

def run(T, intermediate_steps, UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, h, rho, gamma, Xi, Q, c, n_cities, delta_year):

    v = (g*measles_distance_matrix)

    def body(input, t_obs):

        X_t, _, _, _ = input

        pop_index = tf.cast(t_obs/26, dtype = tf.int64)
        pop_t = UKpop[:,pop_index:pop_index+1]

        birth_index = tf.cast(t_obs/26, dtype = tf.int64)
        UKbirths_t  = UKbirths[:,birth_index:(birth_index+1)]

        xi_t = Xi.sample((n_cities, 1))
        q_t  = Q.sample((n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs,:]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        output = body_run(X_t, intermediate_steps, UKbirths_t, pop_t, beta_bar, p, a, is_school_term_array_t, is_start_school_year_array_t_obs, h, rho, gamma, xi_t, q_t, c, n_cities, delta_year, v)

        return output[0], output[1], xi_t, q_t

    X_0 = tfp.distributions.Multinomial( total_count=initial_pop, probs=pi_0).sample()
    Y_0 = tf.zeros((n_cities, 1), dtype=tf.float32)

    X_t, Y_t, Xi_t, Q_t = tf.scan(body, tf.range(0, T+1, dtype=tf.int64), initializer = (X_0, Y_0, Y_0, Y_0)) 

    return X_t, tf.concat((tf.expand_dims(Y_0, axis =0), Y_t), axis =0), Xi_t, Q_t