library(pomp)
library(deSolve)
library(microbenchmark)
source('bsflu_mcmc.R')
source("LNA/LNA_ode_system.R")
source("PAL.R")

################ Simulation study
#Set simulation parameters
init_pop <- c(763-1,1,0)
init_params <- c(2,0.5,0.8, 100)

y <- bsflu$B

ode_step = seq(2, 92, by = 5)
iterations = 500

LNA_time <- function(steps){
  return(1e-9*microbenchmark(SIR_approx_lik_LNA_time(y, init_pop, init_params, steps), times = iterations, unit = "seconds")$time)
}

compare_LNA <- sapply(ode_step, LNA_time)

PAL_time <- function(steps){
  return(1e-9*microbenchmark(SIR_approxlik_R_int(y, init_pop, init_params, steps), times = iterations, unit = "seconds")$time)
}

compare_PAL <- sapply(ode_step, PAL_time)

ratio = (compare_LNA/compare_PAL)


library(matrixStats)

par(cex.main = 1, cex.lab = 1)
plot(ode_step, colMeans(ratio), xlab = "steps", ylab = 'LNA(sec)/PAL(sec)', ty = "l", ylim = c(10, 400) )
polygon(c(ode_step,rev(ode_step)),c(colQuantiles(ratio, probs = 0.75),rev(colQuantiles(ratio, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(ode_step,rev(ode_step)),c(colQuantiles(ratio, probs = 0.95),rev(colQuantiles(ratio, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))

legend("topright", legend=c("Mean ratio", " 25%-75% percentile", " 5%-95% percentile"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(1, 1, 1), lwd = c(4, 8, 10), pch=c(NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.2)


# GGplot
LNA_time <- function(steps){
  return(1e-9*microbenchmark(SIR_approx_lik_LNA_time(y, init_pop, init_params, steps), times = 1, unit = "seconds")$time)
}

PAL_time <- function(steps){
  return(1e-9*microbenchmark(SIR_approxlik_R_int(y, init_pop, init_params, steps), times = 1, unit = "seconds")$time)
}

compare <- data.frame(ode_step = rep(ode_step, iterations * 2), sec = rep(NA, length(ode_step) * iterations * 2), alg = rep("null", length(ode_step) * iterations * 2))

for(reps in 1:iterations){
  
  time = rep(0, length(ode_step))
  for(i in 1:length(ode_step)){
    
    compare[(reps - 1) * (length(ode_step)) + i, 1] <- ode_step[i]
    compare[(reps - 1) * (length(ode_step)) + i, 2] <- 1e-9*microbenchmark(SIR_approx_lik_LNA_time(y, init_pop, init_params, ode_step[i]), times = 1, unit = "seconds")$time
    compare[(reps - 1) * (length(ode_step)) + i, 3] <-  "LNA"
  }
  
}

next_to_fill = (iterations - 1) * (length(ode_step)) + length(ode_step)

for(reps in 1:iterations){
  
  time = rep(0, length(ode_step))
  for(i in 1:length(ode_step)){
    
    compare[(reps - 1) * (length(ode_step)) + i + next_to_fill, 1] <- ode_step[i]
    compare[(reps - 1) * (length(ode_step)) + i + next_to_fill, 2] <- 1e-9*microbenchmark(SIR_approxlik_R_int(y, init_pop, init_params, ode_step[i]), times = 1, unit = "seconds")$time
    compare[(reps - 1) * (length(ode_step)) + i + next_to_fill, 3] <-  "PAL"
  }
  
}

library(tidyverse)

colnames(compare) <- c("steps", "seconds", "algorithm")

ggplot(compare, aes(x = steps, y = seconds, color = algorithm)) + 
  #geom_point() +
  #ylim(-0.001, .002)+
  geom_smooth()

compare_LNA_gg <- matrix(NA, nrow = iterations, ncol = length(ode_step))
compare_PAL_gg <- matrix(NA, nrow = iterations, ncol = length(ode_step))
for(reps in 1:iterations){
  
  time = rep(0, length(ode_step))
  for(i in 1:length(ode_step)){
    
    compare_LNA_gg[reps,i] <- compare[(reps - 1) * (length(ode_step)) + i, 2]
  
    }
  
}

next_to_fill = (iterations - 1) * (length(ode_step)) + length(ode_step)

for(reps in 1:iterations){
  
  time = rep(0, length(ode_step))
  for(i in 1:length(ode_step)){
    
    compare_PAL_gg[reps,i] <- compare[(reps - 1) * (length(ode_step)) + i + next_to_fill, 2] 
    
    }
  
}

ratio_gg = (compare_LNA_gg/compare_PAL_gg)


library(matrixStats)

par(cex.main = 1, cex.lab = 1)
plot(ode_step, colMeans(ratio_gg), xlab = "steps", ylab = 'LNA(sec)/PAL(sec)', ty = "l", ylim = c(50, 600) )
polygon(c(ode_step,rev(ode_step)),c(colQuantiles(ratio_gg, probs = 0.75),rev(colQuantiles(ratio_gg, probs = 0.25))),lty=0,col=rgb(0,0.3,1,0.4))
polygon(c(ode_step,rev(ode_step)),c(colQuantiles(ratio_gg, probs = 0.95),rev(colQuantiles(ratio_gg, probs = 0.05))),lty=0,col=rgb(0,0.3,1,0.2))

legend("topright", legend=c("Mean ratio", " 25%-75% percentile", " 5%-95% percentile"), 
       box.lwd = 1, bty = 'n',bg = "white",
       col=c("black", "blue", rgb(0,0.3,1,0.65), rgb(0,0.3,1,0.2)), lty= c(1, 1, 1), lwd = c(4, 8, 10), pch=c(NA, NA, NA),
       seg.len=0.25, y.intersp=0.65, x.intersp=0.25, cex=1.2)

