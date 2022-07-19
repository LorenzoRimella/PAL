library(Rcpp)
library(RcppArmadillo)
library(mvtnorm)


## Priors
logprior <- function(params){
  return(dmvnorm(params, mean = c(0,0), sigma = diag(c(1,1), nrow = )))
}

logprior_poisson <- function(params){
  return(dmvnorm(params, mean = c(0,0,0.5), sigma = diag(c(1,1,1), nrow = )))
}

## Priors
#logprior_LNA <- function(params){
#  return(dmvnorm(params, mean = c(0,0,0.5,0), sigma = diag(c(1,1,0.5,1), nrow = 4)))
#}

## Priors
logprior_v <- function(params){
  return(dgamma(params, 1.1, (1.1*(5*763/10^5)^2)))
}

logprior_q <- function(params){
  return(dgamma(params, 1.1, (1.1/10)))
}

## gibbs proposal

proposal <- function(par, var, indicator = 0){
  prop <- rnorm(1, par, var)
  if(prop <= 0){prop = par}
  if(prop>=1 && indicator == 1){prop = par}
  return(prop)
}

#proposal <- function(par, var, indicator = 0){
#  prop <- rnorm(1, par, var)
#  if(prop>=1 && indicator == 1){prop = -1}
#  while(prop <= 0){
#    prop <- rnorm(1, par, var)
#    if(prop>=1 && indicator == 1){prop = -1}
#  }
#  return(prop)
#}


## poisson mcmc

poisson_mcmc <- function(y, init_pop, init_params, n_iter, rw_params){
  param_samples <- matrix(nrow = 3, ncol = n_iter+1)
  param_samples[,1] <- init_params
  ind <- c(0,0,1)
  old_Loglik <- SIR_approx_lik(y, init_pop, init_params)
  accepted_params <- c(0,0,0)
  for (i in 1:n_iter) {
    print(i)
    sample <- param_samples[,i]
    for (j in 1:3){
      prop <- sample
      prop[j] <- proposal(param_samples[j,i], rw_params[j], ind[j])
      
      new_Loglik <- SIR_approx_lik(y, init_pop, prop)
      prior_diff <- logprior_poisson(prop) - logprior_poisson(sample)
      Log_lik_diff <- new_Loglik - old_Loglik
      
      u <- runif(1)
      
      if(u < exp(prior_diff + Log_lik_diff)){
        sample <- prop
        old_Loglik <- new_Loglik
        accepted_params[j] = accepted_params[j] + 1 - as.numeric(prop[j]==sample[j]) 
        print('accepted')
        print(sample)
      }
      else{print('rejected')}
    }
    param_samples[,i+1] = sample
  }
  
  acceptance_ratio <- accepted_params/n_iter
  
  out <- list(param_samples = param_samples, acceptance_ratio = acceptance_ratio)
}

## LNA mcmc

LNA_mcmc <- function(y, init_pop, init_params, n_iter, rw_params){
  param_samples <- matrix(nrow = 4, ncol = n_iter+1)
  param_samples[,1] <- init_params
  ind <- c(0,0,1,0)
  old_Loglik <- SIR_approx_lik_LNA(y, init_pop, init_params)
  accepted_params <- c(0,0,0,0)
  for (i in 1:n_iter) {
    print(i)
    sample <- param_samples[,i]
    for (j in 1:4){
      prop <- sample
      prop[j] <- proposal(param_samples[j,i], rw_params[j], ind[j])
      
      new_Loglik <- SIR_approx_lik_LNA(y, init_pop, prop)
      prior_diff <- logprior(prop[1:2]) - logprior(sample[1:2]) + logprior_q(prop[3]) - logprior_q(sample[3]) + logprior_v(prop[4]) - logprior_v(sample[4])
      Log_lik_diff <- new_Loglik - old_Loglik
      
      u <- runif(1)
      
      if(u < exp(prior_diff + Log_lik_diff)){
        old_Loglik <- new_Loglik
        accepted_params[j] = accepted_params[j] + 1- as.numeric(prop[j]==sample[j])
        sample <- prop
        print('accepted')
        print(sample)
      }
      else{print('rejected')}
    }
    param_samples[,i+1] = sample
  }
  
  acceptance_ratio <- accepted_params/n_iter
  
  out <- list(param_samples = param_samples, acceptance_ratio = acceptance_ratio)
}

# LNA 2 parameters
LNA_mcmc_3param <- function(y, init_pop, init_params, n_iter, rw_params){
  param_samples <- matrix(nrow = 3, ncol = n_iter+1)
  param_samples[,1] <- init_params
  ind <- c(0,0,1)
  old_Loglik <- SIR_approx_lik_LNA(y, init_pop, init_params)
  accepted_params <- c(0,0,0)
  for (i in 1:n_iter) {
    print(i)
    sample <- param_samples[,i]
    for (j in 1:3){
      prop <- sample
      prop[j] <- proposal(param_samples[j,i], rw_params[j], ind[j])
      
      new_Loglik <- SIR_approx_lik_LNA(y, init_pop, prop)
      prior_diff <- logprior(prop) - logprior(sample)
      Log_lik_diff <- new_Loglik - old_Loglik
      
      u <- runif(1)
      
      if(u < exp(prior_diff + Log_lik_diff)){
        sample <- prop
        old_Loglik <- new_Loglik
        accepted_params[j] = accepted_params[j] + 1
        print('accepted')
        print(sample)
      }
      else{print('rejected')}
    }
    param_samples[,i+1] = sample
  }
  
  acceptance_ratio <- accepted_params/n_iter
  
  out <- list(param_samples = param_samples, acceptance_ratio = acceptance_ratio)
}


## pmcmc

pmcmc_bsflu <- function(y, init_pop, init_params, n_particles, n_iter, rw_params){
  param_samples <- matrix(nrow = 3, ncol = n_iter+1)
  param_samples[,1] <- init_params
  ind <- c(0,0,1)
  old_Loglik <- Particle_likelihood_SIR(init_pop, y, init_params, n_particles)
  accepted_params <- c(0,0,0)
  for (i in 1:n_iter) {
    print(i)
    sample <- param_samples[,i]
    for (j in 1:3){
      prop <- sample
      prop[j] <- proposal(param_samples[j,i], rw_params[j], ind[j])
      
      new_Loglik <- Particle_likelihood_SIR(init_pop, y, prop, n_particles)
      prior_diff <- logprior_poisson(prop) - logprior_poisson(sample)
      Log_lik_diff <- new_Loglik - old_Loglik
      
      u <- runif(1)
      
      if(u < exp(prior_diff + Log_lik_diff)){
        sample <- prop
        old_Loglik <- new_Loglik
        accepted_params[j] = accepted_params[j] + 1
        print('accepted')
        print(sample)
      }
      else{print('rejected')}
    }
    param_samples[,i+1] = sample
  }
  
  acceptance_ratio <- accepted_params/n_iter
  
  out <- list(param_samples = param_samples, acceptance_ratio = acceptance_ratio)
}





dapmcmc_bsflu <- function(y, init_pop, init_params, n_particles, n_iter, rw_params){
  param_samples <- matrix(nrow = 3, ncol = n_iter+1)
  param_samples[,1] <- init_params
  ind <- c(0,0,1)
  old_Loglik <- SIR_approx_lik(y, init_pop, init_params)
  pfold_Loglik <- Particle_likelihood_SIR(init_pop, y, init_params, n_particles)
  accepted_poi_params <- c(0,0,0)
  accepted_pf_params <- c(0,0,0)
  for (i in 1:n_iter) {
    print(i)
    sample <- param_samples[,i]
    for (j in 1:3){
      prop <- sample
      prop[j] <- proposal(param_samples[j,i], rw_params[j], ind[j])
      
      new_Loglik <- SIR_approx_lik(y, init_pop, prop)
      prior_diff <- logprior_poisson(prop) - logprior_poisson(sample)
      Log_lik_diff <- new_Loglik - old_Loglik
      
      u <- runif(1)
      
      if(u < exp(prior_diff + Log_lik_diff)){
        accepted_poi_params[j] = accepted_poi_params[j] + 1
        print('accepted by poi')
        new_pf_loglik <- Particle_likelihood_SIR(init_pop, y, prop, n_particles)
        
        pfLog_lik_diff <- new_pf_loglik - pfold_Loglik
        
        v <- runif(1)
        if(v < exp( pfLog_lik_diff - Log_lik_diff)){
        sample <- prop
        old_Loglik <- new_Loglik
        pfold_Loglik <- new_pf_loglik
        accepted_pf_params[j] = accepted_pf_params[j] + 1
        print('accepted')
        print(sample)
        }
        else(print('rejected'))
        }
      else{print('rejected')}
    }
    param_samples[,i+1] = sample
  }
  
  accepted_poi_ratio <- accepted_poi_params/n_iter
  accepted_pf_ratio <- accepted_pf_params/n_iter
  
  out <- list(param_samples = param_samples, accepted_poi_ratio = accepted_poi_ratio,accepted_pf_ratio=accepted_pf_ratio )
}

