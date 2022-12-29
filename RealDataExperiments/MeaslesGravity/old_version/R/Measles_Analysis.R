library(Rcpp)
sourceCpp("cpp/measles_sim.cpp")
sourceCpp('cpp/measles_inference.cpp')


# Load data
load(file = 'data/datamat.Rdata')
load("data/UKMeaslesmetadata.Rdata")
## Format births
births <- as.matrix(births, nrow= 468)


## Set parameter values for simulation test

U = 40
params = 4*c(3, 0.2, 0.1)
c = 0.4
a = 0.3
S_0sim=0.032
E_0sim=0.00005
I_0sim=0.00004
m <- 4
t <- 468
g=1000
h=1
init_dist_props1 = rbind(rep(S_0sim,U), rep(E_0sim,U), rep(I_0sim,U), rep(1-S_0sim-E_0sim-I_0sim,U))

# Simulate data

sim <- Sim_Gravity(p,init_dist_props1, v_by_g , births, params, q , survivalprob, t, n_steps = 4 , c  , g, h = 1, a, m = 4, K = U)


### Check that Profile likelihoods maximise in sensible places

betas <- seq(from =8, to =16, length = 50)
lik <- c()
for (i in 1:length(betas)) {
  print(i)
  lik[i] <- PoiGravity(p, t(sim), t(t(init_dist_props1)*p),v_by_g, births, params = c(betas[i],0.8,0.4) , q, survivalprob  , c=0.4, g , h = 1, a = 0.3,m = 4, k = 40, ncores = 4, tstar = 100)
}
par(mfrow = c(1,1))
plot(betas,lik)
abline(v = 12)


## profile lik for rho
rhos <- seq(from =0.1, to =1.5, length = 50)
lik <- c()
for (i in 1:length(rhos)) {
  print(i)
  lik[i] <- PoiGravity(p, t(sim), t(t(init_dist_props1)*p),v_by_g,births, params = c(12,rhos[i],0.4) , q, survivalprob  , c=0.4, g = 1000 , h = 1, a = 0.3,m = 4, k = 40, ncores = 4, tstar = 1)
}
par(mfrow = c(1,1))
plot(rhos,lik)
abline(v = 0.8)

## profile lik for gamma
gammas <- seq(from =0.01, to =0.5, length = 50)
lik <- c()
for (i in 1:length(gammas)) {
  print(i)
  lik[i] <- PoiGravity(p, t(sim), t(t(init_dist_props1)*p),v_by_g, births, params = c(12,0.8,gammas[i]) , q, survivalprob  , c=0.4, g = 1000 , h = 1, a = 0.3,m = 4, k = 40, ncores = 4, tstar = 1)
}
par(mfrow = c(1,1))
plot(gammas,lik)
abline(v = 0.4)

## profile lik for a
as <- seq(from =0.1, to =0.5, length = 50)
lik <- c()
for (i in 1:length(as)) {
  print(i)
  lik[i] <- PoiGravity(p, t(sim), t(t(init_dist_props1)*p),v_by_g, births, params = c(12,0.8,0.4) , q, survivalprob  , c=0.4, g = 1500 , h = 1, as[i],m = 4, k = 40, ncores = 4, tstar = 1)
}
par(mfrow = c(1,1))
plot(as,lik)
abline(v = 0.3)


cs <- seq(from =0.2, to =0.55, length = 50)
lik <- c()
for (i in 1:length(cs)) {
  print(i)
  lik[i] <- PoiGravity(p, t(sim), t(t(init_dist_props1)*p),v_by_g, births, params = c(12,0.8,0.4) , q, survivalprob  , cs[i], g = 1500 , h = 1, a= 0.3,m = 4, k = 40, ncores = 4, tstar = 1)
}
par(mfrow = c(1,1))
plot(cs,lik)
abline(v = 0.4)


gs <- seq(from =500, to = 1500, length = 25)
lik <- c()
for (i in 1:length(gs)) {
  print(i)
  lik[i] <- PoiGravity(p, t(sim), t(t(init_dist_props1)*p),v_by_g, births, params = c(12,0.8,0.4) , q, survivalprob  , c=0.4, gs[i] , h = 1, a = 0.3,m = 4, k = 40, ncores = 4, tstar = 1)
}
par(mfrow = c(1,1))
plot(gs,lik)
abline(v = 1000)


### Example optimisation scheme on real data 

# Initialise parameters
U <- 40
S_0=0.032
E_0=0.00005
I_0=0.00004
init_dist_props1 = rbind(rep(S_0,U), rep(E_0,U), rep(I_0,U), rep(1-S_0-E_0-I_0,U))

betastraj <- c(runif(1,5,50))
rhostraj <-  c(runif(1,0.2,0.8))
gammastraj <-  c(runif(1,0.2,2.8))
atraj <-  c(runif(1,0.2,0.8))
ctraj <-  c(runif(1,0.2,0.8))
gtraj <-  c(runif(1,10,1000))



t_1 = Sys.time()
for (f in 1:500) {
  print(f)
  for (j in 1:15) {
    t1 <- Sys.time()
    ### Optimise beta
    optfunbeta <- function(beta){
      
      -PoiGravity(p, datamat[-469,], t(t(init_dist_props1)*p),v_by_g, births, params = c(beta,rhostraj[length(rhostraj)],gammastraj[length(gammastraj)]) , q, survivalprob  , ctraj[length(ctraj)], g = gtraj[length(gtraj)], h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 100)
    }
    
    optbeta <- optim(betastraj[length(betastraj)], fn = optfunbeta, lower = 0.1, upper = 40, method = "Brent")
    betastraj <- append(betastraj,optbeta$par)
    
    ### Optimise rho
    
    optfunrho <- function(rho){
      - PoiGravity(p, datamat[-469,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rho,gammastraj[length(gammastraj)]) , q, survivalprob  , ctraj[length(ctraj)], g = gtraj[length(gtraj)], h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 100)
    }
    optrho <- optim(rhostraj[length(rhostraj)], fn = optfunrho, lower = 0.1, upper = 1.5, method = "Brent")
    rhostraj <- append(rhostraj,optrho$par)
    
    
    ### Optimise gamma
    
    optfungamma <- function(gamma){
      - PoiGravity(p, datamat[-469,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rhostraj[length(rhostraj)],gamma) , q, survivalprob  , ctraj[length(ctraj)], g = gtraj[length(gtraj)], h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 100)
    }
    optgamma <- optim(gammastraj[length(gammastraj)], fn = optfungamma, lower = 0.1, upper = 0.95, method = "Brent")
    gammastraj <- append(gammastraj,optgamma$par)
    
    ### Optimise a
    
    optfuna <- function(a){
      - PoiGravity(p, datamat[-469,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rhostraj[length(rhostraj)],gammastraj[length(gammastraj)]) , q, survivalprob  , ctraj[length(ctraj)], g = gtraj[length(gtraj)], h = 1, a ,m = 4, k = 40, ncores = 1, tstar = 100)
    }
    opta <- optim(atraj[length(atraj)], fn = optfuna, lower = 0.05, upper = 0.95, method = "Brent")
    atraj <- append(atraj,opta$par)
    
    ## Optimise c
    optfunc <- function(c){
      -PoiGravity(p, datamat[-469,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rhostraj[length(rhostraj)],gammastraj[length(gammastraj)]) , q, survivalprob  , c, g = gtraj[length(gtraj)], h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 100)
    }
    optc <- optim(ctraj[length(ctraj)], fn = optfunc, lower = 0.1, upper = 1, method = "Brent")
    ctraj <- append(ctraj,optc$par)
    
    ## Optimise g
    optfung <- function(g){
      -PoiGravity(p, datamat[-469,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rhostraj[length(rhostraj)],gammastraj[length(gammastraj)]) , q, survivalprob  , ctraj[length(ctraj)], g , h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 100)
    }
    optg <- optim(gtraj[length(gtraj)], fn = optfung, lower = 0, upper = 5000, method = "Brent")
    gtraj <- append(gtraj, optg$par)
    
    t2 <- Sys.time()
    print(t2-t1)
  }
  
  ### Optimise IVPs
  for (i in 1:50) {
    optfunS <- function(S){
      
      init_dist_props1 = rbind(rep(S,U), rep(E_0,U), rep(I_0,U), rep(1-S-E_0-I_0,U))
      -PoiGravity(p, datamat[1:25,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rhostraj[length(rhostraj)],gammastraj[length(gammastraj)]) , q, survivalprob  , ctraj[length(ctraj)], g = gtraj[length(gtraj)], h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 1)
    }
    
    optS <- optim(S_0, fn = optfunS, lower = 0, upper = 0.04, method = "Brent")
    S_0 <- optS$par
    
    optfunE <- function(E){
      
      init_dist_props1 = rbind(rep(S_0,U), rep(E,U), rep(I_0,U), rep(1-S_0-E-I_0,U))
      -PoiGravity(p, datamat[1:25,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rhostraj[length(rhostraj)],gammastraj[length(gammastraj)]) , q, survivalprob  , ctraj[length(ctraj)], g = gtraj[length(gtraj)], h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 1)
    }
    
    optE <- optim(E_0, fn = optfunE, lower = 0, upper = 0.001, method = "Brent")
    E_0 <- optE$par
    
    optfunI <- function(I){
      
      init_dist_props1 = rbind(rep(S_0,U), rep(E_0,U), rep(I,U), rep(1-S_0-E_0-I,U))
      -PoiGravity(p, datamat[1:25,], t(t(init_dist_props1)*p),v_by_g, births, params = c(betastraj[length(betastraj)],rhostraj[length(rhostraj)],gammastraj[length(gammastraj)]) , q, survivalprob  , ctraj[length(ctraj)], g = gtraj[length(gtraj)], h = 1, a = atraj[length(atraj)],m = 4, k = 40, ncores = 1, tstar = 1)
    }
    optI <- optim(I_0, fn = optfunI, lower = 0, upper = 0.001, method = "Brent")
    I_0 <- optI$par
    init_dist_props1 = rbind(rep(S_0,U), rep(E_0,U), rep(I_0,U), rep(1-S_0-E_0-I_0,U))
  }
}
t_2 = Sys.time()

















