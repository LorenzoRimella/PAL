import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def K_eta_SEIR( beta_param, rho, gamma):
    
    def K_eta_matrix(x_t, gamma_noise):
        
        shape_x  =  tuple(x_t.shape[:-1])
        transmission_probability = 1 - tf.math.exp(-beta_param[0]*tf.einsum("...j, ...j -> ...j", x_t[...,2], gamma_noise)/(tf.reduce_sum(x_t, axis = -1)))
        transmission_probability = tf.reshape(transmission_probability, shape_x + (1, 1))

        latent_probability       = (1 - tf.math.exp(-rho))*tf.ones(transmission_probability.shape)

        recovery_probability     = (1 - tf.math.exp(-gamma))*tf.ones(transmission_probability.shape)

        K_eta_h__n_r1 = tf.concat((                            1 - transmission_probability, transmission_probability, tf.zeros(shape_x + (1, 2))), axis = -1)
        K_eta_h__n_r2 = tf.concat((tf.zeros(shape_x + (1, 1)), 1 - latent_probability,       latent_probability,       tf.zeros(shape_x + (1, 1))), axis = -1)
        K_eta_h__n_r3 = tf.concat((tf.zeros(shape_x + (1, 2)), 1 - recovery_probability,     recovery_probability                                ), axis = -1)
        K_eta_h__n_r4 = tf.concat((tf.zeros(shape_x + (1, 3)), tf.ones(shape_x + (1, 1))                                                         ), axis = -1)
        K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = -2)

        return K_eta_h__n

    return K_eta_matrix
   

class Compartmental_model():

    def __init__(self, pi_0, delta, beta_param, rho, gamma, alpha, q, G, kappa):

        self.pi_0  = pi_0

        self.delta = delta

        self.beta_param = beta_param
        self.K_eta = K_eta_SEIR(beta_param, rho, gamma)

        self.alpha   = alpha
        self.q_param = q    
        self.G       = G    
        self.kappa   = kappa

def sim_step_0(self, n, number_simulations):

    x_0 =  (tfp.distributions.Multinomial(total_count = n, probs = tf.squeeze(self.pi_0)).sample(number_simulations))

    return x_0

def sim_step_t(self, n, barx_tm1, K_eta_tm1, number_simulations):
    
    transitions_x = tfp.distributions.Multinomial(total_count = barx_tm1, probs = K_eta_tm1).sample()
    tildex_t      = tf.reduce_sum(transitions_x, axis = 1)

    # births
    births = tfp.distributions.Poisson(n*tf.squeeze(self.alpha)).sample(number_simulations)
    x_t    = tildex_t + births

    return x_t

def sim_observe(self, X):

    # Observed process
    # q = tfp.distributions.Beta(self.q_param[:,0], self.q_param[:,1]).sample((X.shape[0], X.shape[1]))
    q = tfp.distributions.TruncatedNormal(self.q_param[:,0], self.q_param[:,1], 0, 1).sample(X.shape[:-1])
    barY        = tfp.distributions.Binomial(total_count = X, probs = q).sample()

    transitionsY = tfp.distributions.Multinomial(total_count = barY, probs = tf.reshape(self.G, (1, 1, self.G.shape[0], self.G.shape[1]))).sample()
    tildeY      = tf.reduce_sum(transitionsY, axis = 2)

    # clutter
    hatY = tfp.distributions.Poisson(tf.squeeze(self.kappa)).sample((X.shape[0], X.shape[1]))
    Y    = tildeY + hatY

    return Y, q

def sim_observe_Q(self, X, q):

    # Observed process
    barY        = tfp.distributions.Binomial(total_count = X, probs = q).sample()

    transitionsY = tfp.distributions.Multinomial(total_count = barY, probs = tf.reshape(self.G, (1, 1, self.G.shape[0], self.G.shape[1]))).sample()
    tildeY      = tf.reduce_sum(transitionsY, axis = 2)

    # clutter
    hatY = tfp.distributions.Poisson(tf.squeeze(self.kappa)).sample((X.shape[0], X.shape[1]))
    Y    = tildeY + hatY

    return Y, q

def sim_run(self, n, T, number_simulations):

    def body(input, t):

        x_tm1 = input[0]

        # transitions
        gamma_noise   = tfp.distributions.Gamma(1/(self.beta_param[1]), self.beta_param[1]).sample(number_simulations)
        # gamma_noise   = tf.where(tf.math.is_nan(gamma_noise), tf.ones_like(gamma_noise), gamma_noise)
        # gamma_noise   = tf.ones(number_simulations)
        
        barx_tm1  = tfp.distributions.Binomial(total_count = x_tm1, probs = tf.transpose(self.delta)).sample()
        K_eta_tm1     = self.K_eta(barx_tm1, gamma_noise)
        x_t = sim_step_t(self, n, barx_tm1, K_eta_tm1, number_simulations)

        return x_t, gamma_noise

    x_0 = sim_step_0(self, n, number_simulations)
    # gamma_noise   = tfp.distributions.Gamma(-1, -1).sample(self.number_simulations)
    # gamma_noise   = tf.where(tf.math.is_nan(gamma_noise), tf.ones_like(gamma_noise), gamma_noise)
    gamma_noise   = tf.ones(number_simulations)

    X, GAMMA = tf.scan(body, tf.range(0, T), initializer = (x_0, gamma_noise))

    Y, Q = sim_observe(self, X)

    return X, Y, GAMMA, Q

def sim_run_Q(self, n, T, Q, number_simulations=1):

    def body(input, t):

        x_tm1 = input[0]

        # transitions
        # gamma_noise   = tfp.distributions.Gamma(1/(self.beta_param[1]), self.beta_param[1]).sample(number_simulations)
        # gamma_noise   = tf.where(tf.math.is_nan(gamma_noise), tf.ones_like(gamma_noise), gamma_noise)
        gamma_noise   = tf.ones(number_simulations)
        
        barx_tm1  = tfp.distributions.Binomial(total_count = x_tm1, probs = tf.transpose(self.delta)).sample()
        K_eta_tm1     = self.K_eta(barx_tm1, gamma_noise)
        x_t = sim_step_t(self, n, barx_tm1, K_eta_tm1, number_simulations)

        return x_t, gamma_noise

    x_0 = sim_step_0(self, n, number_simulations)
    # gamma_noise   = tfp.distributions.Gamma(-1, -1).sample(self.number_simulations)
    # gamma_noise   = tf.where(tf.math.is_nan(gamma_noise), tf.ones_like(gamma_noise), gamma_noise)
    gamma_noise   = tf.ones(number_simulations)

    X, GAMMA = tf.scan(body, tf.range(0, T), initializer = (x_0, gamma_noise))

    Y, Q = sim_observe_Q(self, X, Q)

    return X, Y, GAMMA, Q
        
class PoissonApproximation():

    def __init__(self, n, pi_0, delta, beta_param, rho, gamma, alpha, q_param, G, kappa):
        
        self.pi_0     = pi_0
        self.lambda_0 = n*pi_0

        self.delta = delta

        self.alpha = alpha
        self.G     = G    
        self.kappa = kappa
        self.n     = n
        self.q_param = q_param

@tf.function(jit_compile=True)
def prediction(n, K_eta_tm1, alpha, barlambda_tm1_death):

    lambda_t  = tf.einsum("...j,...ji->...i", barlambda_tm1_death, K_eta_tm1)+ n*tf.squeeze(alpha)

    return lambda_t

@tf.function(jit_compile=True)
def update(G, kappa, lambda_t, y_t, q_sampled):

    qlambdaGpluskappa     = tf.einsum("...i,ij->...j", q_sampled*lambda_t, G)+ tf.squeeze(kappa)
    maskqlambdaGpluskappa = -10*tf.cast(qlambdaGpluskappa==0, dtype = tf.float32)

    qG = tf.einsum("...i,ji->...ji", q_sampled, G)

    mu_t   = qlambdaGpluskappa

    beforelambda = 1 - q_sampled + tf.einsum("...i,...ij->...j", tf.einsum("...j,...j->...j", y_t, 1/(mu_t + maskqlambdaGpluskappa)), qG)

    barlambda_t = beforelambda*lambda_t

    logw_t = tf.reduce_sum(tfp.distributions.Poisson(mu_t).log_prob(y_t), axis =-1)
    # logw_t = tf.experimental.numpy.nansum(tfp.distributions.Poisson(mu_t).log_prob(y_t), axis =-1)

    return barlambda_t, logw_t

def run_filter(n, pi_0, delta, beta_param, rho, gamma, alpha, G, kappa, Y, GAMMA, Q):

    def body(input, t):

        y_t = Y[t+1,...]
        barlambda_tm1, logw_tm1 = input

        gamma_noise = GAMMA[t+1,...]
        q_sampled = Q[t+1,...]

        K_eta = K_eta_SEIR(beta_param, rho, gamma)
        barlambda_tm1_death = barlambda_tm1*tf.squeeze(delta)
        K_eta_tm1 = K_eta(barlambda_tm1_death, gamma_noise)
        
        lambda_t       = prediction(n, K_eta_tm1, alpha, barlambda_tm1_death)
        barlambda_t, _ = update(G, kappa, lambda_t, y_t, q_sampled)

        return lambda_t, barlambda_t

    lambda_0 = n*pi_0
    barlambda_tm1 = tf.squeeze(lambda_0)*tf.ones(Y.shape[1:])

    Lambda, barLambda = tf.scan(body, tf.range(0, Y.shape[0]-1), initializer = ( barlambda_tm1, barlambda_tm1))

    return Lambda, barLambda

# @tf.function(jit_compile=True)
# def compute_beta_param(self, lambda_t, y_t):

#     q_param = self.q_param

#     beta_a_corre = y_t + tf.einsum("i,...i->...i", q_param[:,0], tf.ones(lambda_t.shape))
#     beta_a_prior =       tf.einsum("i,...i->...i", q_param[:,0], tf.ones(lambda_t.shape))

#     increment = lambda_t - y_t
#     beta_b_corre = tf.einsum("i,...i->...i", q_param[...,1], tf.ones(lambda_t.shape)) + increment
#     beta_b_prior = tf.einsum("i,...i->...i", q_param[...,1], tf.ones(lambda_t.shape)) 

#     beta_a       = beta_a_prior*tf.cast(increment<=0, dtype = tf.float32) + beta_a_corre*tf.cast(increment>0, dtype = tf.float32)
#     beta_b       = beta_b_prior*tf.cast(increment<=0, dtype = tf.float32) + beta_b_corre*tf.cast(increment>0, dtype = tf.float32)

    # return beta_a, beta_b

@tf.function(jit_compile=True)
def parameters_q_Laplace(q_param, lambda_t, y_t):

    fxi_sigma = tf.einsum("j,...j->...j", q_param[:,1]*q_param[:,1], lambda_t)
    b = tf.ones(fxi_sigma.shape)*tf.transpose(q_param[:,0:1]) - fxi_sigma

    mu_r = (b + tf.sqrt(b*b  + 4*tf.ones(fxi_sigma.shape)*y_t*q_param[:,1]*q_param[:,1]))/2
    sigma_r = 1/((tf.ones(fxi_sigma.shape)*y_t)/(tf.where(tf.ones(fxi_sigma.shape)*y_t==0, tf.ones_like(tf.ones(fxi_sigma.shape)*y_t), mu_r*mu_r)) + 1/(tf.ones(fxi_sigma.shape)*tf.transpose(q_param[:,1:]*q_param[:,1:])))

    mu_r = tf.where(tf.ones(fxi_sigma.shape)*tf.transpose(q_param[:,0:1])<-100, -500*tf.ones(fxi_sigma.shape), mu_r)

    return mu_r, tf.sqrt(sigma_r)

def step_t(n, delta, beta_param, rho, gamma, alpha, q_param, G, kappa, barlambda_tm1, y_t):

    # gamma_noise   = tfp.distributions.Gamma(1/beta_param[1], beta_param[1]).sample((barlambda_tm1.shape[:-1]))
    # gamma_noise   = tf.where(tf.math.is_nan(gamma_noise), tf.ones_like(gamma_noise), gamma_noise)
    gamma_noise = tf.ones_like(barlambda_tm1[...,0])

    K_eta = K_eta_SEIR(beta_param, rho, gamma)
    barlambda_tm1_death = barlambda_tm1*tf.squeeze(delta)
    K_eta_tm1 = K_eta(barlambda_tm1_death, gamma_noise)

    lambda_t    = prediction(n, K_eta_tm1, alpha, barlambda_tm1_death)

    # beta_a, beta_b = self.compute_beta_param(lambda_t, y_t)

    mu_r, sigma_r = parameters_q_Laplace(q_param, lambda_t, y_t)

    q_sampled_rv = tfp.distributions.TruncatedNormal(mu_r, sigma_r, 0, 1)

    q_sampled  = q_sampled_rv.sample()

    q_prior_rv = tfp.distributions.TruncatedNormal(tf.ones(q_sampled.shape)*tf.transpose(q_param[:,0:1]), tf.ones(q_sampled.shape)*tf.transpose(q_param[:,1:]), 0, 1)

    barlambda_t, logw_t = update(G, kappa, lambda_t, y_t, q_sampled)

    log_prior = q_prior_rv.log_prob(q_sampled)
    log_propo = q_sampled_rv.log_prob(q_sampled)

    logw_t_prior_proposal = logw_t + tf.reduce_sum(log_prior, axis =-1) - tf.reduce_sum(log_propo, axis =-1)

    return barlambda_t, gamma_noise, q_sampled, logw_t_prior_proposal

@tf.function(jit_compile=True)
def normalize_w(logw_t):

    M = tf.reduce_max(logw_t, axis = -2, keepdims = True)

    w_t = tf.exp(logw_t - M)/tf.reduce_sum(tf.exp(logw_t - M), axis = -2, keepdims=True)

    return w_t

@tf.function(jit_compile=True)
def systematic_resampling(barlambda_t, gamma_noise, q_sampled, w_t, U):

    n_particles = w_t.shape[-2]

    indeces = tf.cast(tf.linspace(0, n_particles-1, n_particles), dtype = tf.float32)
    indeces = tf.einsum("i,...ia->...ia", indeces, tf.ones(w_t.shape))

    Uis = (U + indeces)/n_particles

    zero_and_w_t = tf.concat((tf.zeros(w_t.shape[:-2]+(1)+w_t.shape[-1:]), w_t), axis = -2) 
    cumulative_w_t = tf.cumsum(zero_and_w_t, axis =-2)

    # def gather_by_particle(i):
        
    #     return tf.reduce_sum(tf.cast(Uis[i:(i+1),:]> cumulative_w_t[:-1,:], dtype = tf.int32), axis = -2)-1

    # indeces_run = tf.cast(tf.linspace(0, n_particles-1, n_particles), dtype = tf.int32)  
    # resample = tf.map_fn(gather_by_particle, indeces_run, dtype = tf.int32)
    resample = tf.reduce_sum(tf.cast(Uis>tf.transpose(cumulative_w_t[:-1,:]), dtype = tf.int32), axis =1, keepdims=True)-1

    barlambda_t_resampled = tf.gather(barlambda_t, resample[:,0], axis = 0)
    gamma_noise_resampled = tf.gather(gamma_noise, resample[:,0], axis = 0)
    q_sampled_resampled   = tf.gather(q_sampled,   resample[:,0], axis = 0)

    return barlambda_t_resampled, gamma_noise_resampled, q_sampled_resampled, resample[:,0]

    # @tf.function(jit_compile=True)
    # def systematic_resampling(self, barlambda_t, gamma_noise, q_sampled, w_t, U):

    #     n_particles = w_t.shape[-2]

    #     indeces = tf.cast(tf.linspace(0, n_particles-1, n_particles), dtype = tf.float32)
    #     indeces = tf.einsum("i,...ia->...ia", indeces, tf.ones(w_t.shape))

    #     Uis = (U + indeces)/n_particles

    #     zero_and_w_t = tf.concat((tf.zeros(w_t.shape[:-2]+(1)+w_t.shape[-1:]), w_t), axis = -2) 
    #     cumulative_w_t = tf.cumsum(zero_and_w_t, axis =-2)

    #     def gather_by_particle(i):
            
    #         return tf.reduce_sum(tf.cast(Uis[:,:,i:(i+1),:]> cumulative_w_t[:,:,:-1,:], dtype = tf.int32), axis = -2)-1

    #     indeces_run = tf.cast(tf.linspace(0, n_particles-1, n_particles), dtype = tf.int32)  

    #     resample = tf.einsum("pij...->ijp...", tf.map_fn(gather_by_particle, indeces_run, dtype = tf.int32))

    #     indeces_run = tf.cast(tf.linspace(0, resample.shape[-1]-1, resample.shape[-1]), dtype = tf.int32)

    #     indeces_param1 = tf.cast(tf.linspace(0, resample.shape[0]-1, resample.shape[0]), dtype = tf.int32)

    #     def gather_param_1(i):

    #         indeces_param2 = tf.cast(tf.linspace(0, resample.shape[1]-1, resample.shape[1]), dtype = tf.int32)

    #         def gather_param_2(j):

    #             indeces_sim = tf.cast(tf.linspace(0, resample.shape[-1]-1, resample.shape[-1]), dtype = tf.int32)

    #             def gather_sim(k):

    #                 barlambda_t_resampled = tf.gather(barlambda_t[i,j,:,k,:], resample[i,j,:,k], axis = 0)
    #                 gamma_noise_resampled = tf.gather(gamma_noise[i,j,:,k,:], resample[i,j,:,k], axis = 0)
    #                 q_sampled_resampled   = tf.gather(q_sampled[i,j,:,k,:], resample[i,j,:,k], axis = 0)
                    
    #                 return barlambda_t_resampled, gamma_noise_resampled, q_sampled_resampled

    #             return tf.einsum("k...j->...kj", tf.map_fn(gather_sim, indeces_sim, dtype = tf.float32))

    #         return tf.map_fn(gather_param_2, indeces_param2, dtype = tf.float32)   

    #     barlambda_t_resamp =  tf.map_fn(gather_param_1, indeces_param1, dtype = tf.float32) 

    #

def run_SMC(n, pi_0, delta, beta_param, rho, gamma, alpha, q_param, G, kappa, n_particles, Y):

    def body(input, t):

        y_t = Y[t+1:t+2,...]
        barlambda_tm1, _, _, w_t, _ = input
        
        barlambda_t, gamma_noise, q_sampled, logw_t = step_t(n, delta, beta_param, rho, gamma, alpha, q_param, G, kappa, barlambda_tm1, y_t)

        w_t = normalize_w(logw_t)

        U = tfp.distributions.Uniform(0, 1).sample(w_t.shape[:-2]+(1)+w_t.shape[-1:])

        barlambda_t_resampled, gamma_noise_resampled, q_sampled_resampled, ancestors = systematic_resampling(barlambda_t, gamma_noise, q_sampled, w_t, U)

        return barlambda_t_resampled, gamma_noise_resampled, q_sampled_resampled , w_t, ancestors

    lambda_0 = n*pi_0
    barlambda_tm1 = tf.squeeze(lambda_0)*tf.ones((n_particles, Y.shape[1], 1))

    initializer_0 = ( barlambda_tm1, -tf.ones(barlambda_tm1.shape[:-1]), -tf.ones(barlambda_tm1.shape), -tf.ones(barlambda_tm1.shape[:-1]), -tf.ones(n_particles, dtype = tf.int32))

    barLambda, GAMMA, Q, W, ancestors = tf.scan(body, tf.range(0, Y.shape[0]-1), initializer = initializer_0)

    return barLambda, GAMMA, Q, W, ancestors

@tf.function(jit_compile=True)
def indeces_computation_(U_i, cumulative_w_t):
    
    return tf.reduce_sum(tf.cast(U_i> cumulative_w_t[:-1,:], dtype = tf.int32), axis = -2)-1

def multi_experiments_systematic_resampling(barlambda_t, gamma_noise, q_sampled, w_t, U):

    def body(input, t):
        
        return systematic_resampling(barlambda_t[:,t:t+1,:], gamma_noise[:,t:t+1], q_sampled[:,t:t+1,:], w_t[:,t:t+1], U[:,t:t+1])

    t = 0
    initializer_0 = systematic_resampling(barlambda_t[:,t:t+1,:], gamma_noise[:,t:t+1], q_sampled[:,t:t+1,:], w_t[:,t:t+1], U[:,t:t+1])

    output = tf.scan(body, tf.range(0, barlambda_t.shape[1]), initializer = initializer_0)

    barlambda_t_resampled, gamma_noise_resampled, q_sampled_resampled = output

    return tf.einsum("ij...->ji...", barlambda_t_resampled[:,:,0,...]), tf.einsum("ij...->ji...", gamma_noise_resampled[:,:,0,...]), tf.einsum("ij...->ji...", q_sampled_resampled[:,:,0,...])

def run_SMC_loglikelihood(n, pi_0, delta, beta_param, rho, gamma, alpha, q_param, G, kappa, n_particles, Y):

    def body(input, t):

        y_t = Y[t+1:t+2,...]
        barlambda_tm1, _, _, w_t = input
        
        barlambda_t, gamma_noise, q_sampled, logw_t = step_t(n, delta, beta_param, rho, gamma, alpha, q_param, G, kappa, barlambda_tm1, y_t)

        w_t = normalize_w(logw_t)

        U = tfp.distributions.Uniform(0, 1).sample(w_t.shape[:-2]+(1)+w_t.shape[-1:])

        barlambda_t_resampled, gamma_noise_resampled, q_sampled_resampled = multi_experiments_systematic_resampling(barlambda_t, gamma_noise, q_sampled, w_t, U)

        return barlambda_t_resampled, gamma_noise_resampled, q_sampled_resampled , logw_t

    lambda_0 = n*pi_0
    barlambda_tm1 = tf.squeeze(lambda_0)*tf.ones((n_particles, Y.shape[1], 1))

    initializer_0 = ( barlambda_tm1, -tf.ones(barlambda_tm1.shape[:-1]), -tf.ones(barlambda_tm1.shape), -tf.ones(barlambda_tm1.shape[:-1]))

    barLambda, _, Q, logW = tf.scan(body, tf.range(0, Y.shape[0]-1), initializer = initializer_0)

    return logW

 


