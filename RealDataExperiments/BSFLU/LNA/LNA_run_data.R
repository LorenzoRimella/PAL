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
init_real <- c(762,1,0)
y <- bsflu$B
