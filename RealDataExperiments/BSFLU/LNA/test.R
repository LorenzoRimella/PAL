# Check the validity of the generated paths
source("LNA_ode_system.R")
library(deSolve)
library(mvtnorm)

n = 1000 
parameters = array(c(1.2/n, 0.6), dim =c(2,1))

times_grid = seq(0, 1, by = 0.1)
time_steps = 20

sample_path = array(NA, dim = c(20, 2, 10))

for (k in 1:10){
  x_prev = n*array(c(0.95,0.05, 0,0, 0,0,0,0), dim = c(8,1)) 
  
  for (i in 1:time_steps){
    
    x_next = array(ode(x_prev, times_grid, LNA_restart, parameters)[11,2:9], dim = c(8, 1))
    sim    = rmvnorm(1, (x_next[1:2] + x_next[3:4]), matrix(x_next[5:8], ncol =2))
    
    sample_path[i,,k] = sim
    x_prev = array(c(sim, x_next[3:4], array(0, dim = c(4))), dim = c(8,1))
  }
}


plot( sample_path[,1,1],                         col = "green", type = "l", lwd = 2, ylim = c(0, n))
lines(sample_path[,2,1],                         col = "red",   type = "l", lwd = 2)
lines(n - sample_path[,1,1] - sample_path[,2,1], col = "blue",  type = "l", lwd = 2)

for (k in 2:10){
  lines(sample_path[,1,k],                         col = "green", type = "l", lwd = 2, ylim = c(0, n))
  lines(sample_path[,2,k],                         col = "red",   type = "l", lwd = 2)
  lines(n - sample_path[,1,k] - sample_path[,2,k], col = "blue",  type = "l", lwd = 2)
}



# test likelihood computation on flu data
library(pomp)

y = bsflu$B
init_pop = c(762,1,0)
pop_size = sum(init_pop)

init_params = c(2/pop_size, 0.5, 0.8, 5)

loglikelihood = SIR_approx_lik_LNA(y, init_pop, init_params)
filter        = SIR_approx_filter_LNA(y, init_pop, init_params)

mean = filter[[1]]
plot( mean[,1],                         col = "green", type = "l", lwd = 2, ylim = c(0, pop_size))
lines(mean[,2],                         col = "red",   type = "l", lwd = 2)
lines(pop_size - mean[,1] - mean[,2], col = "blue",  type = "l", lwd = 2)
points(y)
