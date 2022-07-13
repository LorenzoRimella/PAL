library(Rcpp)
library(pomp)
library(deSolve)
sourceCpp('cpp/SIR_simulator.cpp')
sourceCpp('cpp/Approx_SIR_filter.cpp')
sourceCpp('cpp/SIR_particle_filter.cpp')
source('bsflu_mcmc.R')
source("LNA/LNA_ode_system.R")

################ Simulation study
#Set simulation parameters
init <- c(763-1,1,0)
pars <- c(2,0.5,0.8)
parnames <- c(expression(beta),expression(gamma),expression(q))
colours <- c('darkred', 'darkgreen', 'orange')
t <- 14

## Run simulation or load data used in the paper, delete # as desired
# sim <- SIR_simulator(t, init, pars)
load('data/SIRsim500k')



### Run and time PALMH on simulated data
#t1poi <- Sys.time()
#mcmc_chain <- poisson_mcmc(sim, init, c(2,0.5,0.8), 500000, rw_params = c(0.5,0.05,0.1))
#t2poi <- Sys.time()

# or load pre run data 
load('data/poimcmcsynth500k.Rdata')
#Check acceptance ratio
# mcmc_chain$acceptance_ratio


### Run and time PMCMC
### Particle mcmc
t1pmcmc <- Sys.time()
pmcmc_chain <- pmcmc_bsflu(sim, init, c(2,0.5,0.8), 1000, 10,  rw_params = c(0.5,0.1,0.22))
t2pmcmc <- Sys.time()
## alternatively load a pre run mcmc
load('data/pmcmc500kbsflu.Rdata')


##### time and run daPMCMC
#t1da <- Sys.time()
#dapmcmc_chain <- dapmcmc_bsflu(sim, init, c(2,0.2,0.8), 1000, 500000,  rw_params = 0.85*c(0.4,0.065,0.11))
#t2da <- Sys.time()
#dapmcmc_chain$accepted_poi_ratio
#dapmcmc_chain$accepted_pf_ratio
## or Just load the pre run sample
load('data/dapmcmc500kbsflu.Rdata')


##SYNTHETIC DATA DIAGNOSTIC PLOTS
par(mfrow = c(3,3), cex.lab = 1.4)
## Posterior histograms
par(mfrow = c(3,3))
indices <- sample(100000:500000, 100000)
for (i in 1:3) {
  hist(mcmc_chain$param_samples[i,indices],  main = 'PALMH', xlab = parnames[i], breaks = 27, col = colours[i])
  hist(dapmcmc_chain$param_samples[i,indices], main = 'daPMMH', xlab = parnames[i], breaks = 27, col = colours[i])
  hist(pmcmc_chain$param_samples[i,indices],  main = 'PMMH', xlab = parnames[i], breaks = 27, col = colours[i])
}



# Traceplots
for (i in 1:3) {
  ts.plot(mcmc_chain$param_samples[i,300000:400000],  main = 'PALMH', ylab = '', xlab = 'iteration',  col = colours[i])
  title(ylab = parnames[i], line = 2)
  ts.plot(dapmcmc_chain$param_samples[i,300000:400000], main = 'daPMMH', ylab = '', xlab = 'iteration', col = colours[i])
  title(ylab = parnames[i], line = 2)
  ts.plot(pmcmc_chain$param_samples[i,300000:400000],  main = 'PMMH', ylab = '', xlab = 'iteration', col = colours[i])
  title(ylab = parnames[i], line = 2)
}






# autocorrelation plots
par(mfrow = c(3,3), cex.main = 1.4,cex.lab = 1)
acf(mcmc_chain$param_samples[1,300000:400000],  main = expression('PALMH ACF for ' ~ beta), col = colours[1])
acf(dapmcmc_chain$param_samples[1,300000:400000],  main = expression('daPMMH ACF for ' ~ beta), col = colours[1])
acf(pmcmc_chain$param_samples[1,300000:400000],  main = expression('PMMH ACF for ' ~ beta), col = colours[1])


acf(mcmc_chain$param_samples[2,300000:400000],  main = expression('PALMH ACF for ' ~ rho), col = colours[2])
acf(dapmcmc_chain$param_samples[2,300000:400000],  main = expression('daPMMH ACF for ' ~ rho), col = colours[2])
acf(pmcmc_chain$param_samples[2,300000:400000],  main = expression('PMMH ACF for ' ~ rho), col = colours[2])

acf(mcmc_chain$param_samples[3,300000:400000],  main = expression('PALMH ACF for ' ~ gamma), col = colours[3])
acf(dapmcmc_chain$param_samples[3,300000:400000],  main = expression('daPMMH ACF for ' ~ gamma), col = colours[3])
acf(pmcmc_chain$param_samples[3,300000:400000],  main = expression('PMMH ACF for ' ~ gamma), col = colours[3])





################ REAL DATA
## Set initial distribution and load real data:
init_real <- c(762,1,0)
y <- bsflu$B

## Run mcmc or load pre run chains

#t1 = Sys.time()
#mcmc_chain_real <- LNA_mcmc_3param(y, init_real, c(2,0.5,0.8), 10, rw_params = c(0.4,0.05,0.07))
#t2 = Sys.time()
#
t1 = Sys.time()
mcmc_chain_real <- LNA_mcmc(y, init_real, c(2,0.5,0.8,1), 100000, rw_params = c(0.4,0.05,0.1,10))
t2 = Sys.time()
t2-t1

mcmc_chain_real$acceptance_ratio

ts.plot(mcmc_chain_real$param_samples[1,])
hist(mcmc_chain_real$param_samples[1,])
ts.plot(mcmc_chain_real$param_samples[2,])
hist(mcmc_chain_real$param_samples[2,])
ts.plot(mcmc_chain_real$param_samples[3,])
hist(mcmc_chain_real$param_samples[3,])
ts.plot(mcmc_chain_real$param_samples[4,])
hist(mcmc_chain_real$param_samples[4,])

save(mcmc_chain_real, file = "LNA.RData")
#
#mcmc_chain_real <- poisson_mcmc(y, init_real, c(2,0.5,0.8), 500000, rw_params = c(0.4,0.05,0.1))
#
#pmcmc_chain_real_vague <- pmcmc_bsflu(y, init_real, c(2,0.5,0.8), 1000, 500000,  rw_params = 1.5*c(0.4,0.05,0.1))
#
#dapmcmc_chain_real_vague <- dapmcmc_bsflu(y, init_real, c(2,0.5,0.8), 1000, 500000,  rw_params = 1.5*c(0.4,0.05,0.1))



load('data/poimcmcreal500k.Rdata')
load('data/dapmcmcvague500k.Rdata')
load('data/pmcmcvague500k.Rdata')



### real data diagnostic plots

par(mfrow = c(3,3), cex.lab = 1.3)

for (i in 1:3) {
  hist(mcmc_chain_real$param_samples[i,indices],  main = 'PALMH', xlab = parnames[i], breaks = 27, col = colours[i])
  hist(dapmcmc_chain_real_vague$param_samples[i,indices], main = 'daPMMH', xlab = parnames[i], breaks = 27, col = colours[i])
  hist(pmcmc_chain_real_vague$param_samples[i,indices],  main = 'PMMH', xlab = parnames[i], breaks = 27, col = colours[i])
}


for (i in 1:3) {
  ts.plot(mcmc_chain_real$param_samples[i,300000:400000],  main = 'PALMH', ylab = '', xlab = 'iteration',  col = colours[i])
  title(ylab = parnames[i], line = 2)
  ts.plot(dapmcmc_chain_real_vague$param_samples[i,300000:400000], main = 'daPMMH', ylab = '', xlab = 'iteration', col = colours[i])
  title(ylab = parnames[i], line = 2)
  ts.plot(pmcmc_chain_real_vague$param_samples[i,300000:400000],  main = 'PMMH', ylab = '', xlab = 'iteration', col = colours[i])
  title(ylab = parnames[i], line = 2)
  }

par(mfrow = c(3,3), cex.main = 1.5, cex.lab = 1)
  acf(mcmc_chain_real$param_samples[1,300000:400000],  main = expression('PALMH ACF for ' ~ beta), col = colours[1])
  acf(dapmcmc_chain_real_vague$param_samples[1,300000:400000],  main = expression('daPMMH ACF for ' ~ beta), col = colours[1])
  acf(pmcmc_chain_real_vague$param_samples[1,300000:400000],  main = expression('PMMH ACF for ' ~ beta), col = colours[1])
  
  
  acf(mcmc_chain_real$param_samples[2,300000:400000],  main = expression('PALMH ACF for ' ~ rho), col = colours[2])
  acf(dapmcmc_chain_real_vague$param_samples[2,300000:400000],  main = expression('daPMMH ACF for ' ~ rho), col = colours[2])
  acf(pmcmc_chain_real_vague$param_samples[2,300000:400000],  main = expression('PMMH ACF for ' ~ rho), col = colours[2])
  
  acf(mcmc_chain_real$param_samples[3,300000:400000],  main = expression('PALMH ACF for ' ~ gamma), col = colours[3])
  acf(dapmcmc_chain_real_vague$param_samples[3,300000:400000],  main = expression('daPMMH ACF for ' ~ gamma), col = colours[3])
  acf(pmcmc_chain_real_vague$param_samples[3,300000:400000],  main = expression('PMMH ACF for ' ~ gamma), col = colours[3])







