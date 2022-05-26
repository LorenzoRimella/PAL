library(Rcpp)
library(pomp)
library(matrixStats)
sourceCpp('cpp/SIR_simulator.cpp')
# initial distribution 
init <- c(763-1,1,0)
# load data
y <- bsflu$B
# load posterior samples
load('data/poimcmcreal500k.Rdata')
load('data/dapmcmcvague500k.Rdata')
load('data/pmcmcvague500k.Rdata')
# indices to sample from the posterior
indices <- sample(2e5:5e5, size = 1e4)

### Sample from posterior
poi_sample <- mcmc_chain_real$param_samples[,indices]

par(mfrow= c(2,1))
## Sample from the model
trajectories <- matrix(data= NA, nrow = 1e4,ncol = 14)
for(i in 1:10000) {
  pars <- poi_sample[,i]
  ppsim <- SIR_simulator(14, init, pars)
  trajectories[i,] <- ppsim
}
par(cex.main = 1, cex.lab = 1)
ts.plot(colMeans(trajectories),ylim = c(0, max(trajectories)), ylab = 'Current infected', main = "Posterior predictive plots with PALMH sample")
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.75),rev(colQuantiles(trajectories, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.95),rev(colQuantiles(trajectories, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(y, pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.2)


# repeat for dapmmh sample
dapmcmc_sample <- dapmcmc_chain_real_vague$param_samples[,indices]

trajectories <- matrix(data= NA, nrow = 1e4,ncol = 14)
for(i in 1:10000) {
  pars <- dapmcmc_sample[,i]
  ppsim <- SIR_simulator(14, init, pars)
  trajectories[i,] <- ppsim
}

ts.plot(colMeans(trajectories),ylim = c(0, max(trajectories)), ylab = 'Current infected', main = "Posterior predictive plots with daPMMH sample")
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.75),rev(colQuantiles(trajectories, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:14,rev(1:14)),c(colQuantiles(trajectories, probs = 0.9),rev(colQuantiles(trajectories, probs = 0.1))),lty=0,col=rgb(0,0.3,1,0.2))
points(y, pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.2)


