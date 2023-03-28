library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
sourceCpp('rotavirus_qonly.cpp')
load('real_rotavirus_metadata.Rdata')
prop <- 420

#   lik[i] <-  rotavirus_SMC_obs(init_dist,realdat , 9, c(11.6,0.24,0.34,0.14,0.13),norm_par = c(0.07,0.095), gamma_par = c(3,  1/3),t(prop), 1000,1)$log_lik

# Get the list of arguments passed to R
args <- commandArgs(T)

# Get the first argument, make it numeric because by default it is read as a string
job_id <- as.numeric(args[1])

initial_guess <- c(runif(1,9,18),runif(1,0.05,0.5),runif(1,0.2,1),runif(1,0.05,0.4),runif(1,0.05,0.4),runif(1,0.05,0.2))

astep <- c(0.01,0.001,0.001,0.001,0.001,0.001)
cstep <- 5*c(0.01,0.001,0.001,0.001,0.001,0.001)

Coordinate_ascent_algorithm2 <- function(a,c,init_params, n_steps){
  tstart <- Sys.time()
  par <- init_params
  traj <- matrix(nrow = 6, ncol= n_steps+1)
  lik <- c()
  lik[1] <- rotavirus_SMC_obs(init_dist,realdat , 9, c(par[1],par[2],par[3],par[4],par[5]),norm_par = c(0.07,par[6]), gamma_par = c(420,  1/420),t(prop), 1000,1)$log_lik
  for (i in 1:n_steps) {
    t1 <- Sys.time()
    print(paste('iteration =',i))
    for (j in 1:6) {
      print(j)
      #Delta <- sample(c(-1,1),1)
      Delta <- 1
      par_pos <- par
      par_neg <- par
      par_pos[j] <- par[j]+cstep[j]
      par_neg[j] <- par[j]-cstep[j]
      
      grad_pos <- rotavirus_SMC_obs(init_dist,realdat , 9, c(par_pos[1],par_pos[2],par_pos[3],par_pos[4],par_pos[5]),norm_par = c(0.07,par_pos[6]), gamma_par = c(par_pos[7],  1/par_pos[7]),t(prop), 1000,1)$log_lik
      grad_neg <- rotavirus_SMC_obs(init_dist,realdat , 9, c(par_neg[1],par_neg[2],par_neg[3],par_neg[4],par_neg[5]),norm_par = c(0.07,par_neg[6]), gamma_par = c(par_neg[7],  1/par_neg[7]),t(prop), 1000,1)$log_lik
      print(grad_pos)
      print(grad_neg)
      grad_est <- sign(grad_pos - grad_neg)
      print(grad_est)
      par[j] <- par[j] + astep[j]*grad_est + runif(1,-1,1)*0.1*astep[j]
      if(par[j]==0){par[j] <- 0.01}
    }
    
    traj[,i+1] = par
    lik[i+1] <- rotavirus_SMC_obs(init_dist,realdat , 9, c(par[1],par[2],par[3],par[4],par[5]),norm_par = c(0.07,par[6]), gamma_par = c(420,  1/420),t(prop), 1000,1)$log_lik
    t2 <- Sys.time()
    print(par)
    print(lik[i+1])
    print(t2-t1)
  }
  tend <- Sys.time()
  time <- tstart - tend
  output <- list(traj = traj, lik =lik, time=time)
  return(output)
}

sign_ascent_qonly <- Coordinate_ascent_algorithm2(a,c,initial_guess, 5000)

filename <- paste('sign_ascent_qonly',job_id, '.Rdata', sep = '')

save(sign_ascent_qonly , file = filename)






