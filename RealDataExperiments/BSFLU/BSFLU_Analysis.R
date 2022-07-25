library(Rcpp)
library(pomp)
library(deSolve)
sourceCpp('cpp/SIR_simulator.cpp')
sourceCpp('cpp/Approx_SIR_filter.cpp')
sourceCpp('cpp/SIR_particle_filter.cpp')
source('bsflu_mcmc.R')
source("LNA/LNA_ode_system.R")
source("PAL.R")

################ Simulation study
#Set simulation parameters
init_pop <- c(763-1,1,0)
init_params <- c(2,0.5,0.8, 400)

y <- bsflu$B

ode_step = seq(2, 50, by = 1)
iterations = 10

compare = matrix(NA, nrow = iterations , ncol = length(ode_step))

for(reps in 1:iterations){
  
  time = rep(0, length(ode_step))
  for(i in 1:length(ode_step)){
    
    start = Sys.time()
    a = SIR_approx_lik_LNA_time(y, init_pop, init_params, ode_step[i])
    time[i] = as.double(Sys.time()-start)
    
  }
  
  compare[reps,] = time
  
}

compare_PAL = matrix(NA, nrow = iterations , ncol = 1)

for(reps in 1:iterations){
  
  start = Sys.time()
  a = SIR_approxlik_R(y, init_pop, init_params)
  compare_PAL[reps] = as.double(Sys.time()-start)

  
}

plot(ode_step, compare[1,], type = "l", ylim = c(0.0, 0.11))

for(reps in 2:10){
  lines(ode_step, compare[reps,], type = "l")
  abline(h=compare_PAL[reps])
}

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
mcmc_chain_real <- LNA_mcmc(y, init_real, c(2,0.5,0.8,10), 100000, rw_params = c(0.4,0.05,0.1,500))
t2 = Sys.time()
t2-t1

mcmc_chain_real$acceptance_ratio

dev.off()
par(mar =c(1,5,1,1), mfrow = c(4,1))
ts.plot(mcmc_chain_real$param_samples[1,], ylab="beta")
ts.plot(mcmc_chain_real$param_samples[2,], ylab="gamma")
ts.plot(mcmc_chain_real$param_samples[3,], ylab="q")
ts.plot(mcmc_chain_real$param_samples[4,], ylab="v")

scatter.smooth(mcmc_chain_real$param_samples[3,],mcmc_chain_real$param_samples[4,], ty="p")


par(mfrow = c(4,1))
acf(mcmc_chain_real$param_samples[1,])
acf(mcmc_chain_real$param_samples[2,])
acf(mcmc_chain_real$param_samples[3,])
acf(mcmc_chain_real$param_samples[4,])


par( mfrow = c(4,1))
hist(mcmc_chain_real$param_samples[1,], freq = F, breaks =100)
hist(mcmc_chain_real$param_samples[2,], freq = F, breaks =100)
hist(mcmc_chain_real$param_samples[3,], freq = F, breaks =100)
hist(mcmc_chain_real$param_samples[4,], freq = F, breaks =100)

plot(seq(0,10,by=0.1), 2*dnorm(seq(0,10,by=0.1), 0, 1))

#save(mcmc_chain_real, file = "LNA.RData")
#
#mcmc_chain_real <- poisson_mcmc(y, init_real, c(2,0.5,0.8), 500000, rw_params = c(0.4,0.05,0.1))
#
#pmcmc_chain_real_vague <- pmcmc_bsflu(y, init_real, c(2,0.5,0.8), 1000, 500000,  rw_params = 1.5*c(0.4,0.05,0.1))
#
#dapmcmc_chain_real_vague <- dapmcmc_bsflu(y, init_real, c(2,0.5,0.8), 1000, 500000,  rw_params = 1.5*c(0.4,0.05,0.1))

## Run mcmc or load pre run chains
t1 <- Sys.time()
mcmc_chain_real <- LNA_mcmc_3param(y, init_real, c(2,0.5,0.8), 100000, rw_params = c(0.4,0.05,0.075))
t2 <- Sys.time()

mcmc_chain_real$acceptance_ratio

ts.plot(mcmc_chain_real$param_samples[1,])
ts.plot(mcmc_chain_real$param_samples[2,])
ts.plot(mcmc_chain_real$param_samples[3,])
ts.plot(mcmc_chain_real$param_samples[4,])

save(mcmc_chain_real, file = "LNA.RData")
#
#mcmc_chain_real <- poisson_mcmc(y, init_real, c(2,0.5,0.8), 500000, rw_params = c(0.4,0.05,0.1))
#
#pmcmc_chain_real_vague <- pmcmc_bsflu(y, init_real, c(2,0.5,0.8), 1000, 500000,  rw_params = 1.5*c(0.4,0.05,0.1))
#
#dapmcmc_chain_real_vague <- dapmcmc_bsflu(y, init_real, c(2,0.5,0.8), 1000, 500000,  rw_params = 1.5*c(0.4,0.05,0.1))

par(mfrow = c(3,1))
ts.plot(mcmc_chain_real$param_samples[1,])
ts.plot(mcmc_chain_real$param_samples[2,])
ts.plot(mcmc_chain_real$param_samples[3,])
plot(mcmc_chain_real$param_samples[1,],mcmc_chain_real$param_samples[3,])
acf(mcmc_chain_real$param_samples[3,])
mcmc_chain_real$acceptance_ratio


### LNA posterior predictive

indices <- sample(1000:100000, replace = T, size = 20000)
trajectories <- matrix(data= NA, nrow = 2e3,ncol = 14)
j=0
for(i in indices) {
  j=j+1
  print(j)
  pars <- mcmc_chain_real$param_samples[,i]
  filt <- SIR_approx_filter_LNA(y, init, pars)
  trajectories[j,] <- rmvnorm(1,mean = pars[3]*filt$mean_time[,2], sigma = diag((pars[3]^2)*filt$Sigma_time[,2,2]^2  + filt$mean_time[,2]*pars[3]*(1-pars[3])))
}
par(mfrow = c(1,1))
par(cex.main = 1, cex.lab = 1)
ts.plot(colMeans(trajectories),ylim = c(0, max(trajectories)), ylab = 'Current infected', main = "Posterior predictive plots with LNA sample")
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.75),rev(colQuantiles(trajectories, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.95),rev(colQuantiles(trajectories, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(y, pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.2)

### LNA posterior predictive with binom stochastic model

par(mfrow= c(1,1))
## Sample from the model
indices <- sample(10000:100000, replace = T, size = 1000)
trajectories <- matrix(data= NA, nrow = 1e3,ncol = 14)
j=0
init = c(762,1,0)
for(i in indices) {
  j=j+1
  i = indices[j]
  print(j)
  pars <- mcmc_chain_real$param_samples[,i]
  ppsim <- SIR_simulator_LNA(14, init, pars)
  trajectories[j,] <- ppsim
}
y <- bsflu$B
#library(MatrixGenerics)
library(matrixStats)
dev.off()
par(cex.main = 1, cex.lab = 1)
ts.plot(colMeans(trajectories),ylim = c(0, 500), ylab = 'Current infected', main = "Posterior predictive plots with PALMH sample")
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.75),rev(colQuantiles(trajectories, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.95),rev(colQuantiles(trajectories, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(y, pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.2)


hist(mcmc_chain_real$param_samples[4,], freq = F, breaks = 100)
lines(seq(0.01,4000,by=0.5), dnorm(seq(0.01,4000,by=0.5), 400, 300), ty = "l")

lines(seq(0.01,4000,by=0.5), dgamma(seq(0.01,4000,by=0.5), 1.1, 1.1*(5*763/10^5)^2), ty = "l")
plot(seq(0.01,100,by=0.5), 2*dnorm(seq(0.01,100,by=0.5), 0,1), ty = "l")


plot(seq(0.01,400,by=0.5), dgamma(seq(0.01,400,by=0.5), 1.1, 1.1*(5*763/10^5)^2), ty = "l")

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



  test <- data.frame(q = mcmc_chain_real$param_samples[3,], v = mcmc_chain_real$param_samples[4,])
  library(ggplot2)
  ggplot(test, aes(x = q, y = v)) + stat_bin_2d()
  stat_bin2d
  plt = ggplot(test, aes(x = q, y = v)) + stat_density_2d_filled() +  ylim(0.0,1000)
  library(pomp)
  bsflu$B



