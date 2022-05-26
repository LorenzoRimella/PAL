library(rstan)
library(cmdstanr)
library(bayesplot)
library(matrixStats)
# Load flu data and ODE meta data
load('stan_d_ode')
## Define model using stan file
model <- cmdstan_model('stan/flu.stan')
# run model
fit <- model$sample(data = stan_d, chains = 1, iter_warmup = 100000, iter_sampling = 500000)

## Posterior histograms and traceplots
indices <- sample(100000:500000, size = 10000)
par(mfrow = c(3,4), cex.lab = 1.45)
paramnamesode <- c(expression(q^ODE),expression(beta[11]^ODE),expression(beta[12]^ODE),expression(beta[13]^ODE),expression(beta[14]^ODE),expression(beta[22]^ODE),expression(beta[23]^ODE),expression(beta[24]^ODE),expression(beta[33]^ODE),expression(beta[34]^ODE),expression(beta[44]^ODE))
hist(fit$draws(variables = 'rho')[indices],freq = F, xlab = paramnamesode[1], main = '', breaks = 20)
for (i in 1:10) {
  hist(fit$draws(variables = paste('params[', i,  ']', sep=''))[indices],freq = F, xlab = paramnamesode[i+1], main = '', breaks = 20)
}
par(mfrow = c(3,4), cex.lab = 1.5)
ts.plot(c(fit$draws(variables = 'rho')), type = 'l', ylab = '', main = '', xlab = 'Iteration')
title(ylab = paramnamesode[1], line = 1.8, cex.lab = 1.5)
for (i in 1:10) {
  ts.plot(c(fit$draws(variables = paste('params[', i,  ']', sep=''))), ylab = '', xlab = 'Iteration', main = '')
  title(ylab = paramnamesode[i+1], line = 1.8, cex.lab = 1.5)
}

#### Posterior predictive checks

nsamples = 10000
indices <- sample(100000:500000, nsamples, replace = T)
group1 <- matrix(nrow = nsamples, ncol = 19)
group2 <- matrix(nrow = nsamples, ncol = 19)
group3 <- matrix(nrow = nsamples, ncol = 19)
group4 <- matrix(nrow = nsamples, ncol = 19)
rates1 <- fit$draws(variables = 'incidence1')
rates2 <- fit$draws(variables = 'incidence2') 
rates3 <- fit$draws(variables = 'incidence3') 
rates4 <- fit$draws(variables = 'incidence4') 
j=0
for(i in indices) {
  j=j+1
  group1[j,] <- rpois(rep(1,19), c(0,as.numeric(rates1[i,1,])))
  group2[j,] <- rpois(rep(1,19), c(0,as.numeric(rates2[i,1,])))
  group3[j,] <- rpois(rep(1,19), c(0,as.numeric(rates3[i,1,])))
  group4[j,] <- rpois(rep(1,19), c(0,as.numeric(rates4[i,1,])))
}


par(mfrow = c(2,2))

ts.plot(colMeans(group1),ylim = c(0, 100), ylab = 'New reported cases', main = "Age 00-04")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.75),rev(colQuantiles(group1, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.975),rev(colQuantiles(group1, probs = 0.025))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,1], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)

ts.plot(colMeans(group2),ylim = c(0, 330), ylab = 'New reported cases', main = "Age 05-14")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.75),rev(colQuantiles(group2, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.975),rev(colQuantiles(group2, probs = 0.025))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,2], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group3),ylim = c(0, 300), ylab = 'New reported cases', main = "Age 15-44")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.75),rev(colQuantiles(group3, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.975),rev(colQuantiles(group3, probs = 0.025))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,3], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group4),ylim = c(0, 80), ylab = 'New reported cases', main = "Age 45+")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.75),rev(colQuantiles(group4, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.975),rev(colQuantiles(group4, probs = 0.025))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,4], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)


### Calculate R0 posterior using next generation matrix

R0ODE <- c()
vec1 <- c(949/8000)*c(fit$draws(variables = paste('params[', 1,  ']', sep='')))
vec2 <- c(949/8000)*c(fit$draws(variables = paste('params[', 2,  ']', sep='')))
vec3 <- c(1690/8000)*c(fit$draws(variables = paste('params[', 2,  ']', sep='')))
vec4 <- c(949/8000)*c(fit$draws(variables = paste('params[', 3,  ']', sep='')))
vec5 <- c(3467/8000)*c(fit$draws(variables = paste('params[', 3,  ']', sep='')))
vec6 <- c(949/8000)*c(fit$draws(variables = paste('params[', 4,  ']', sep='')))
vec7 <- c(1894/8000)*c(fit$draws(variables = paste('params[', 4,  ']', sep='')))
vec8 <- c(1690/8000)*c(fit$draws(variables = paste('params[', 5,  ']', sep='')))
vec9 <- c(1690/8000)*c(fit$draws(variables = paste('params[', 6,  ']', sep='')))
vec10 <- c(3467/8000)*c(fit$draws(variables = paste('params[', 6,  ']', sep='')))
vec11<- c(1690/8000)*c(fit$draws(variables = paste('params[', 7,  ']', sep='')))
vec12 <- c(1894/8000)*c(fit$draws(variables = paste('params[', 7,  ']', sep='')))
vec13 <- c(3467/8000)*c(fit$draws(variables = paste('params[', 8,  ']', sep='')))
vec14 <- c(3467/8000)*c(fit$draws(variables = paste('params[', 9,  ']', sep='')))
vec15 <- c(1894/8000)*c(fit$draws(variables = paste('params[', 9,  ']', sep='')))
vec16 <- c(1894/8000)*c(fit$draws(variables = paste('params[', 10,  ']', sep='')))



for (i in 1:7500) {
  next_gen_mat <- matrix(data = 0, nrow = 4, ncol = 4)
  print(i)
  next_gen_mat[1,1] <- vec1[i]
  next_gen_mat[1,2] <- vec2[i]
  next_gen_mat[2,1] <- vec3[i]
  next_gen_mat[1,3] <- vec4[i]
  next_gen_mat[3,1] <- vec5[i]
  next_gen_mat[1,4] <- vec6[i]
  next_gen_mat[4,1] <- vec7[i]
  next_gen_mat[2,2] <- vec8[i]
  next_gen_mat[2,3] <- vec9[i]
  next_gen_mat[3,2] <- vec10[i]
  next_gen_mat[2,4] <- vec11[i]
  next_gen_mat[4,2] <- vec12[i]
  next_gen_mat[3,3] <- vec13[i]
  next_gen_mat[3,4] <- vec14[i]
  next_gen_mat[4,3] <- vec15[i]
  next_gen_mat[4,4] <- vec16[i]
  
  R0ODE[i] <- max(eigen(next_gen_mat*(1.5/7))$values)
}
par(mfrow=c(1,2), cex.lab = 1)

hist(R0ODE, main = '', breaks = 25, xlab = expression(R[0]^ODE), freq = F, ylim = c(0,20))




