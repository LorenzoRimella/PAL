// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <math.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export(name=SIR_simulator)]]
vec SIR_simulator(int t,
                  vec init_dist,
                  vec params){
  
  mat state(3,t+1);
  int n = sum(init_dist);
  state.col(0) = init_dist;
  vec obs(t);
  int new_infs;
  int new_rem;
  
  
  
  for(int i=0; i<t; i++){
    
    new_infs = R::rbinom(state(0,i), 1 - exp(-params(0)*state(1,i)/n));
    new_rem = R::rbinom(state(1,i), 1 - exp(-params(1)));
    
    state(0,i+1) = state(0,i) - new_infs;
    state(1,i+1) = state(1,i) + new_infs - new_rem;
    state(2,i+1) = state(2,i) + new_rem;
    
    obs(i) = R::rbinom(state(1,i+1), params(2));
  }
  
  return(obs);
  
}
