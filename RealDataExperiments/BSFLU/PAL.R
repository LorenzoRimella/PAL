SIR_approxlik_R <- function(y,pop,pars){
  lambda  <-  pop
  n = sum(pop)
  K = matrix(data = 0, nrow = 3, ncol = 3)
  K[2,3] = 1 - exp(-pars[2])
  K[2,2] = exp(-pars[2])
  K[3,3] = 1
  lik = c()
  for(i in 1:length(y)){
    
    K[1,2] = 1- exp(-pars[1]*(lambda[2]/n))
    K[1,1] = exp(-pars[1]*(lambda[2]/n))
    
    lambda_ <- (lambda)%*%K
    lambda <- lambda_
    lambda[2] <- y[i] + (1-pars[3])*lambda_[2]
      
    lik[i] = dpois(y[i],lambda_[2]*pars[3], log = T)
  }
  
  return(sum(lik))
}
