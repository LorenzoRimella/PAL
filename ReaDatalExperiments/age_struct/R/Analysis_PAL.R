library(rstan)

## Load data
dat <- read.csv('data/flu_data_1957.csv')
# remove nuiscance line
dat <- as.matrix(dat[,-1])
stan_d <- list(y = dat)

# Run stan HMC
paramnames <- c(expression(q),expression(beta[11]),expression(beta[12]),expression(beta[13]),expression(beta[14]),expression(beta[22]),expression(beta[23]),expression(beta[24]),expression(beta[33]),expression(beta[34]),expression(beta[44]))
outputreal_stochastic_1 <- stan('stan/SEIR_AGE_FLU_daily.stan', data = stan_d, warmup = 100000,  iter = 500000, cores = 4, chains = 1)

## Posterior estimates and traceplots
par(mfrow = c(3,4), cex.lab=1.45)
indices <- sample(100000:500000, size = 10000)
stochsample_1 <- extract(outputreal_stochastic_1)
hist(stochsample_1$q[indices], breaks = 20, freq = F, xlab = paramnames[1], main = '')
for (i in 1:10) {
  hist(stochsample_1$params[indices,i], freq = F, breaks = 20, xlab = paramnames[i+1], main = '')
}

par(mfrow = c(3,4))
ts.plot(stochsample_1$q, main = '', ylab = '', xlab = 'iteration')
title(ylab = paramnames[1], line = 1.8, cex.lab = 1.5)
for (i in 1:10) {
  ts.plot(stochsample_1$params[,i], main = '', ylab = '', xlab = 'iteration')
  title(ylab = paramnames[1+i], line = 1.8, cex.lab = 1.5)
}


## Posterior predictive plots 
## You may need to restart R to laod rcpp functions. Rcpp and Stan do not get along.
library(Rcpp)
library(matrixStats)
sourceCpp('cpp/simulator/simulator_correct.cpp')
stochsamp <- extract(outputreal_stochastic_1)

#Load approximate posterior sample
init <- matrix(c(949,0,1,0,1689,0,1,0,3466,0,1,0,1893,0,1,0), nrow =4)
nsamples = 5000
group1 <- matrix(nrow = nsamples, ncol = 19)
group2 <- matrix(nrow = nsamples, ncol = 19)
group3 <- matrix(nrow = nsamples, ncol = 19)
group4 <- matrix(nrow = nsamples, ncol = 19)
for(i in 1:5000){
  pars <- stochsamp$params[2000+i,]
  ppsim <- SEIR_daily_sim(init_pop = init, params = c(pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],pars[7],pars[8],pars[9],pars[10]),q = stochsamp$q[2000+i], h=1)
  group1[i,] <- ppsim[1,]
  group2[i,] <- ppsim[2,]
  group3[i,] <- ppsim[3,]
  group4[i,] <- ppsim[4,]
}


par(mfrow = c(2,2))

ts.plot(colMeans(group1),ylim = c(0, 100), ylab = 'New reported cases', main = "Age 00-04")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.75),rev(colQuantiles(group1, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.95),rev(colQuantiles(group1, probs = 0.15))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,1], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)

ts.plot(colMeans(group2),ylim = c(0, 270), ylab = 'New reported cases', main = "Age 05-14")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.75),rev(colQuantiles(group2, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.95),rev(colQuantiles(group2, probs = 0.15))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,2], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group3),ylim = c(0, 250), ylab = 'New reported cases', main = "Age 15-44")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.75),rev(colQuantiles(group3, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.95),rev(colQuantiles(group3, probs = 0.15))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,3], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group4),ylim = c(0, 60), ylab = 'New reported cases', main = "Age 45+")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.75),rev(colQuantiles(group4, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.95),rev(colQuantiles(group4, probs = 0.15))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,4], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)







## Estimate R_0 with next gen matrix

next_gen_mat <- matrix(data = 0, nrow = 4, ncol = 4)

next_gen_mat[1,1] <- (949/8000)*mean(stochsample_1$params[,1])
next_gen_mat[1,2] <- (949/8000)*mean(stochsample_1$params[,2])
next_gen_mat[2,1] <- (1690/8000)*mean(stochsample_1$params[,2])
next_gen_mat[1,3] <- (949/8000)*mean(stochsample_1$params[,3])
next_gen_mat[3,1] <- (3467/8000)*mean(stochsample_1$params[,3])
next_gen_mat[1,4] <- (949/8000)*mean(stochsample_1$params[,4])
next_gen_mat[4,1] <- (1894/8000)*mean(stochsample_1$params[,4])
next_gen_mat[2,2] <- (1690/8000)*mean(stochsample_1$params[,5])
next_gen_mat[2,3] <- (1690/8000)*mean(stochsample_1$params[,6])
next_gen_mat[3,2] <- (3467/8000)*mean(stochsample_1$params[,6])
next_gen_mat[2,4] <- (1690/8000)*mean(stochsample_1$params[,7])
next_gen_mat[4,2] <- (1894/8000)*mean(stochsample_1$params[,7])
next_gen_mat[3,3] <- (3467/8000)*mean(stochsample_1$params[,8])
next_gen_mat[3,4] <- (3467/8000)*mean(stochsample_1$params[,9])
next_gen_mat[4,3] <- (1894/8000)*mean(stochsample_1$params[,9])
next_gen_mat[4,4] <- (1894/8000)*mean(stochsample_1$params[,10])

eigen(next_gen_mat*(1.5/7))

### Posterior R0
R0stoch <- c()
for (i in 1:7500) {
  next_gen_mat[1,1] <- (949/8000)*(stochsample_1$params[i,1])
  next_gen_mat[1,2] <- (949/8000)*(stochsample_1$params[i,2])
  next_gen_mat[2,1] <- (1690/8000)*(stochsample_1$params[i,2])
  next_gen_mat[1,3] <- (949/8000)*(stochsample_1$params[i,3])
  next_gen_mat[3,1] <- (3467/8000)*(stochsample_1$params[i,3])
  next_gen_mat[1,4] <- (949/8000)*(stochsample_1$params[i,4])
  next_gen_mat[4,1] <- (1894/8000)*(stochsample_1$params[i,4])
  next_gen_mat[2,2] <- (1690/8000)*(stochsample_1$params[i,5])
  next_gen_mat[2,3] <- (1690/8000)*(stochsample_1$params[i,6])
  next_gen_mat[3,2] <- (3467/8000)*(stochsample_1$params[i,6])
  next_gen_mat[2,4] <- (1690/8000)*(stochsample_1$params[i,7])
  next_gen_mat[4,2] <- (1894/8000)*(stochsample_1$params[i,7])
  next_gen_mat[3,3] <- (3467/8000)*(stochsample_1$params[i,8])
  next_gen_mat[3,4] <- (3467/8000)*(stochsample_1$params[i,9])
  next_gen_mat[4,3] <- (1894/8000)*(stochsample_1$params[i,9])
  next_gen_mat[4,4] <- (1894/8000)*(stochsample_1$params[i,10])
  
  R0stoch[i] <- max(eigen(next_gen_mat*(1.5/7))$values)
}
par(mfrow = c(1,1))
hist(R0stoch,breaks = 20, ylim = c(0,35), main = '', xlab = expression(R[0]^PAL), freq = F)

