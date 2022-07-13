library(pomp)

################ Simulation study
#Set simulation parameters
init <- c(763-1,1,0)
pars <- c(2,0.5,0.8)
parnames <- c(expression(beta),expression(gamma),expression(q))
colours <- c('darkred', 'darkgreen', 'orange')
t <- 14

################ REAL DATA
## Set initial distribution and load real data:
init_pop <- c(762,1,0)
load('data/SIRsim500k')
y <- sim

gamma_grid = seq(0.1, 1, length=50)
lik = c()
for(i in 1:length(gamma_grid)){
  #print(i)
init_params <- array(c(2,gamma_grid[i],0.8))
 lik[i] = SIR_approx_lik_LNA(y, init_pop, init_params)
}
plot(gamma_grid,lik)


beta_grid = seq(1, 3, length=50)
lik = c()
for(i in 1:length(beta_grid)){
  #print(i)
  init_params <- array(c(beta_grid[i], 0.5,0.8,1))
  lik[i] = SIR_approx_lik_LNA(y, init_pop, init_params)
}
plot(beta_grid,lik)


q_grid = seq(0.6, 1.5, length=50)
lik = c()
for(i in 1:length(q_grid)){
  #print(i)
  init_params = array(c(2, 0.5, q_grid[i],10))
  lik[i] = SIR_approx_lik_LNA(y, init_pop, init_params)
}
plot(q_grid,lik)


v_grid = seq(100, 1000, length=50)
lik = c()
for(i in 1:length(v_grid)){
  #print(i)
  init_params = array(c(2, 0.5, 0.8, v_grid[i]))
  lik[i] = SIR_approx_lik_LNA(y, init_pop, init_params)
}
plot(v_grid,lik)
