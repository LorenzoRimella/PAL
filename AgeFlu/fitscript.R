library(rstan)
library(Rcpp)
library(RcppArmadillo)
library(matrixStats)
library(bayesplot)
rstan_options(javascript = FALSE)
setwd("~/Documents/PhD/multinomialapproxproj/R/SEIR_AGE_FLU")
sourceCpp('Simulator/simulator.cpp')

paramnames <- c(expression(q),expression(beta[11]),expression(beta[12]),expression(beta[13]),expression(beta[14]),expression(beta[22]),expression(beta[23]),expression(beta[24]),expression(beta[33]),expression(beta[34]),expression(beta[44]))
## Initial state of model
init <- matrix(c(947,1,1,0,1688,1,1,0,3465,1,1,0,1892,1,1,0), nrow =4)

## Load data
dat <- read.csv('data/flu_data_1957.csv')
# remove nuiscance line
dat <- as.matrix(dat[,-1])
## Run this line to run stan
#outputreal <- stan('Stan/SEIR_AGE_FLU_daily.stan', data = stan_d, iter = 500000, cores = 4, chains = 2)
#traceplot(outputreal, pars = c('q','params'))
# Or load our run
load('data/500kstochastic')
fitarray <- extract(outputreal)
arraystoch <- array(0,dim = c(11,500000))
#Rescale to account for typo made in error
arraystoch[1,] <- fitarray$q*100
arraystoch[2:11,] <- t(fitarray$params)
par(mfrow = c(3,4), cex.lab = 1.5)
#ts.plot(arraystoch[1,], ylab = '', xlab = 'Iteration')
#title(ylab = paramnames[1], line = 1)
for (i in 1:11) {
  par( cex.axis = 1.5)
  ts.plot(arraystoch[i,100000:110000], ylab = '', xlab = 'Iteration')
  title(ylab = paramnames[i], line = 2.1, cex.lab = 2)
}

par(mfrow = c(3,4), cex.lab = 1.5)
#ts.plot(arraystoch[1,], ylab = '', xlab = 'Iteration')
#title(ylab = paramnames[1], line = 1)
for (i in 1:11) {
  par( cex.lab = 2, cex.axis=1.3)
  hist(arraystoch[i,100000:125000], xlab = paramnames[i], main ='', ylab = '', breaks = 20)
}



# Perform the same analysis for the deterministic model
#outputreal_d <- stan('Stan/SEIR_AGE_FLU_deterministic.stan', data = stan_d, iter = 500000, cores = 4, chains = 2)


## Alternatively load the prerun chains 
load('data/500kdeterministic')
## traceplots
#traceplot(outputreal, pars = c('q','params'))
fitarray <- extract(outputreal_d)
arraydet <- array(0,dim = c(11,500000))

arraydet[1,] <- fitarray$q
arraydet[2:11,] <- t(fitarray$params)
par(mfrow = c(3,4), cex.lab = 1.5)
#ts.plot(arraystoch[1,], ylab = '', xlab = 'Iteration')
#title(ylab = paramnames[1], line = 1)
for (i in 1:11) {
  par( cex.axis = 1.5)
  ts.plot(arraydet[i,100000:110000], ylab = '', xlab = 'Iteration')
  title(ylab = paramnames[i], line = 2.1, cex.lab = 2)
}

par(mfrow = c(3,4), cex.lab = 1.5)
#ts.plot(arraystoch[1,], ylab = '', xlab = 'Iteration')
#title(ylab = paramnames[1], line = 1)
for (i in 1:11) {
  par( cex.lab = 2)
  hist(arraydet[i,100000:125000], xlab = paramnames[i], main ='', ylab = '', breaks = 20)
}



