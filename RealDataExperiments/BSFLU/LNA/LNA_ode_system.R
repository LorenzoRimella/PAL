LNA_restart <- function(t, state, parameters){
  
  eta   = state[1:2]
  m     = state[3:4]
  Sigma = state[5:8]
  Sigma = aperm(array(Sigma, dim = c(2,2)))
  
  beta_param  = parameters[1]
  gamma_param = parameters[2]
  
  S = eta[1]
  I = eta[2]
  
  At = aperm(array(c(-1, 0, 1, -1), dim = c(2,2)))
  h  = array(c(beta_param*S*I, gamma_param*I), dim = c(2,1))
  
  F_ = aperm(array(c(-beta_param*I, -beta_param*S, beta_param*I, (beta_param*S) - gamma_param), dim = c(2,2)))
  H  = matrix(0, ncol=2, nrow=2)
  diag(H) = h
  
  deta_dt = At %*% h
  
  dm_dt   = F_ %*% m
  
  dPsi_dt = (Sigma %*% t(F_)) + (F_ %*% Sigma) + (At %*% (H %*% t(At)))
  
  return(list(c(deta_dt, dm_dt, array(t(dPsi_dt), dim = c(4)))))
}


SIR_approx_lik_LNA <- function(y, init_pop, init_params){
  
  times_grid = seq(0, 1, by = 0.1)
  time_steps = length(y)
  
  x_prev     = array(c(init_pop[1:2], 0,0, 0,0,0,0), dim = c(8,1))
  parameters = init_params[1:2]
  parameters[1] = parameters[1]/763
  q          = init_params[3]
  v          = init_params[4]
  
  loglikelihood = 0 
  
  for (i in 1:time_steps){
    
    x_next = array(ode(x_prev, times_grid, LNA_restart, parameters)[11,2:9], dim = c(8, 1))
    mu_t    = x_next[1:2]
    Sigma_t = matrix(x_next[5:8], ncol = 2)
    
    # alternative:v = mu_t[2]*q*(1-q)
    
    mu_l   = q*mu_t[2]
    Sigma_l= q*q*Sigma_t[2,2] + v
    
    loglikelihood = loglikelihood + dnorm(y[i], mean = mu_l, sd = sqrt(Sigma_l), log = T)
    
    covSigma = array(c(q*Sigma_t[1,2], q*Sigma_t[2,2]), dim = c(2,1))
    
    mu_star    = mu_t    + ((y[i] - mu_l)/(Sigma_l))*covSigma
    Sigma_star = Sigma_t - (1/(Sigma_l))*(covSigma %*% t(covSigma))
    
    x_prev = array(c(mu_star, 0,0, array(Sigma_star, dim = c(4))), dim = c(8, 1))
  }
  
  return(loglikelihood)
  
}


SIR_simulator_LNA <- function(t, init_pop, init_params){
  
  times_grid = seq(0, 1, by = 0.1)
  time_steps = t
  x_prev     = array(c(init_pop[1:2], 0,0, 0,0,0,0), dim = c(8,1))
  parameters = init_params[1:2]
  parameters[1] = parameters[1]/763
  q          = init_params[3]
  v          = init_params[4]
  
  
  y_sim <- c()
  x_sim <- matrix(data = NA, nrow = 2, t)
  for (i in 1:time_steps){
    
    x_next = array(ode(x_prev, times_grid, LNA_restart, parameters)[11,2:9], dim = c(8, 1))
    mu_t    = x_next[1:2]
    Sigma_t = matrix(x_next[5:8], ncol = 2)
    x_sim[,i] = c(-1,-1)
    while(any(x_sim[,i]<c(0,0))) {
      x_sim[,i] <- rmvnorm(1,mean = mu_t, sigma = Sigma_t)
    }
    
    
    # alternative:v = mu_t[2]*q*(1-q)
    
    mu_l   = q*x_sim[2,i]
    Sigma_l= v #x_sim[2,i]*q*(1-q)
    
    y_sim[i] <- rnorm(1,mean = mu_l, sd = sqrt(Sigma_l))
    
    mu_star    = x_sim[,i]
    Sigma_star = Sigma_t
    
    x_prev = array(c(x_sim[,i], x_next[3:4], array(0, dim = c(4))), dim = c(8, 1))
  }
  
  
  return(y_sim)
  
}



SIR_approx_filter_LNA <- function(y, init_pop, init_params){
  
  times_grid = seq(0, 1, by = 0.1)
  time_steps = length(y)
  
  x_prev     = array(c(init_pop[1:2], 0,0, 0,0,0,0), dim = c(8,1))
  parameters = init_params[1:2]
  q          = init_params[3]
  #v          = init_params[4]
  
  mean_time  = array(NA, dim = c(time_steps, 2))
  Sigma_time = array(NA, dim = c(time_steps, 2, 2))
  
  for (i in 1:time_steps){
    
    x_next = array(ode(x_prev, times_grid, LNA_restart, parameters)[11,2:9], dim = c(8, 1))
    mu_t    = x_next[1:2]
    Sigma_t = matrix(x_next[5:8], ncol = 2)
    
    # alternative:
    v = mu_t[2]*q*(1-q)
    
    mu_l   = q*mu_t[2]
    Sigma_l= q*q*Sigma_t[2,2] + v
    
    covSigma = array(c(q*Sigma_t[1,2], q*Sigma_t[2,2]), dim = c(2,1))
    
    mu_star    = mu_t    + ((y[i] - mu_l)/(Sigma_l))*covSigma
    Sigma_star = Sigma_t - (1/(Sigma_l))*(covSigma %*% t(covSigma))
    
    mean_time[i,]   = mu_star
    Sigma_time[i,,] = Sigma_star
    
    x_prev = array(c(mu_star, 0,0, array(Sigma_star, dim = c(4))), dim = c(8, 1))
  }
  #output = dict()
  #output[["mean"]]  = mean_time
  #output[["Sigma"]] = Sigma_time
  
  return(list(mean_time, Sigma_time))
  
}
