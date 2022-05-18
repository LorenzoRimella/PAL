library(Rcpp)
library(matrixStats)
sourceCpp('Simulator/simulator.cpp')
init <- matrix(c(947,1,1,0,1688,1,1,0,3465,1,1,0,1892,1,1,0), nrow =4)

#Load data
dat <- read.csv('data/flu_data_1957.csv')
dat <- as.matrix(dat[,-1])

#Load approximate posterior sample
load('data/thinnedsample_stochastic')
nsamples = 10000
group1 <- matrix(nrow = nsamples, ncol = 19)
group2 <- matrix(nrow = nsamples, ncol = 19)
group3 <- matrix(nrow = nsamples, ncol = 19)
group4 <- matrix(nrow = nsamples, ncol = 19)
for(i in 1:10000) {
  pars <- stoch_sample$params[i,]
  ppsim <- SEIR_daily_sim(init_pop = init, params = c(pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],pars[7],pars[8],pars[9],pars[10]),q = stoch_sample$q[i], h=1)
  group1[i,] <- ppsim[1,]
  group2[i,] <- ppsim[2,]
  group3[i,] <- ppsim[3,]
  group4[i,] <- ppsim[4,]
}


par(mfrow = c(2,2))

ts.plot(colMeans(group1),ylim = c(0, 100), ylab = 'New reported cases', main = "Age 00-04")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.75),rev(colQuantiles(group1, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.95),rev(colQuantiles(group1, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,1], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)

ts.plot(colMeans(group2),ylim = c(0, 250), ylab = 'New reported cases', main = "Age 05-14")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.75),rev(colQuantiles(group2, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.95),rev(colQuantiles(group2, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,2], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group3),ylim = c(0, 250), ylab = 'New reported cases', main = "Age 15-44")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.75),rev(colQuantiles(group3, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.95),rev(colQuantiles(group3, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,3], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group4),ylim = c(0, 60), ylab = 'New reported cases', main = "Age 45+")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.75),rev(colQuantiles(group4, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.95),rev(colQuantiles(group4, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,4], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)



# Load approximate posterior sample
load('data/thinnedsample_deterministic')

group1 <- matrix(nrow = nsamples, ncol = 19)
group2 <- matrix(nrow = nsamples, ncol = 19)
group3 <- matrix(nrow = nsamples, ncol = 19)
group4 <- matrix(nrow = nsamples, ncol = 19)
for(i in 1:10000) {
  pars <- det_sample$params[i,]
  ppsim <- SEIR_daily_sim_det(init_pop = init, params = c(pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],pars[7],pars[8],pars[9],pars[10]),q = det_sample$q[i], h=1)
  group1[i,] <- ppsim[1,]
  group2[i,] <- ppsim[2,]
  group3[i,] <- ppsim[3,]
  group4[i,] <- ppsim[4,]
}


par(mfrow = c(2,2))

ts.plot(colMeans(group1),ylim = c(0, 100), ylab = 'New reported cases', main = "Age 00-04")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.75),rev(colQuantiles(group1, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group1, probs = 0.95),rev(colQuantiles(group1, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,1], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)

ts.plot(colMeans(group2),ylim = c(0, 250), ylab = 'New reported cases', main = "Age 05-14")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.75),rev(colQuantiles(group2, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group2, probs = 0.95),rev(colQuantiles(group2, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,2], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group3),ylim = c(0, 250), ylab = 'New reported cases', main = "Age 15-44")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.75),rev(colQuantiles(group3, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group3, probs = 0.95),rev(colQuantiles(group3, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,3], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)
ts.plot(colMeans(group4),ylim = c(0, 60), ylab = 'New reported cases', main = "Age 45+")
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.75),rev(colQuantiles(group4, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(1:19,rev(1:19)),c(colQuantiles(group4, probs = 0.95),rev(colQuantiles(group4, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))
points(dat[,4], pch = 19)
legend("topright", legend=c(" Data", " Posterior predictive mean", " 50% credible interval", " 90% credible interval"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(NA, 1, 1, 1), lwd = c(NA, 2, 8, 10), pch=c(19, NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.4)



