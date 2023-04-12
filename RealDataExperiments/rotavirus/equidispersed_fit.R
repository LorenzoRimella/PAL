library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
sourceCpp('equidispersed_model.cpp')
load('real_rotavirus_metadata.Rdata')

# Get the list of arguments passed to R
args <- commandArgs(T)

# Get the first argument, make it numeric because by default it is read as a string
job_id <- as.numeric(args[1])

initial_guess <- c(runif(1,9,18),runif(1,0.05,0.5),runif(1,0.2,1),runif(1,0.05,0.4),runif(1,0.05,0.4))

astep <- 0.5*c(0.01,0.001,0.001,0.001,0.001,0.001)
cstep <- c(0.01,0.001,0.001,0.001,0.001,0.001)

Coordinate_ascent_algorithm2 <- function(a,c,init_params, n_steps){
  tstart <- Sys.time()
  par <- init_params
  traj <- matrix(nrow = 5, ncol= n_steps+1)
  lik <- c()
  lik[1] <- rotavirus_equidispersed(init_dist, realdat, 9, c(par[1],par[2],par[3],par[4],par[5]),0.07)$log_lik
  # lik[1] <- rotavirus_SMC_obs(init_dist,realdat , 9, c(par[1],par[2],par[3],par[4],par[5]),norm_par = c(0.07,par[6]), gamma_par = c(420,  1/420),t(prop), 1000,1)$log_lik
  for (i in 1:n_steps) {
    if(i%%100==0){print(paste('iteration =',i))}
    t1 <- Sys.time()
    
    for (j in 1:5) {
      # print(j)
      #Delta <- sample(c(-1,1),1)
      Delta <- 1
      par_pos <- par
      par_neg <- par
      par_pos[j] <- par[j]+cstep[j]
      par_neg[j] <- par[j]-cstep[j]
      
      grad_pos <- rotavirus_equidispersed(init_dist, realdat, 9,  c(par_pos[1],par_pos[2],par_pos[3],par_pos[4],par_pos[5]),0.07)$log_lik 
      grad_neg <- rotavirus_equidispersed(init_dist, realdat, 9,  c(par_neg[1],par_neg[2],par_neg[3],par_neg[4],par_neg[5]),0.07)$log_lik 
      # print(grad_pos)
      # print(grad_neg)
      grad_est <- sign(grad_pos - grad_neg)
      # print(grad_est)
      if(i>7500){par[j] <- par[j] + (1/(i-7500)^1.01)*astep[j]*grad_est + (1/(i-7500)^1.01)*runif(1,-1,1)*0.1*astep[j]}
      else{par[j] <- par[j] + astep[j]*grad_est + runif(1,-1,1)*0.1*astep[j]}
      if(par[j]==0){par[j] <- 0.01}
    }
    
    traj[,i+1] = par
    lik[i+1] <- rotavirus_equidispersed(init_dist, realdat, 9, c(par[1],par[2],par[3],par[4],par[5]),0.07)$log_lik
    t2 <- Sys.time()
  }
  tend <- Sys.time()
  time <- tstart - tend
  output <- list(traj = traj, lik =lik, time=time)
  return(output)
}

sign_ascent <- Coordinate_ascent_algorithm2(a,c,initial_guess, 10000)


filename <- paste('sign_ascent_equidispersed',job_id, '.Rdata', sep = '')

save(sign_ascent, file = filename)





