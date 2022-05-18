// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <math.h>
using namespace Rcpp;
using namespace arma;


// [[Rcpp::export(name=SIR_approx_lik)]]
double SIR_approx_lik(vec y,
                      vec init_dist,
                      vec params){
  
  int t = y.size();
  int n = sum(init_dist);
  mat lambda_(3,t);
  mat lambda(3,t+1);
  mat K(3,3, fill::zeros);
  vec w(t, fill::zeros);
  lambda.col(0) = init_dist;
  
  K(1,2) = 1 - exp(-params[1]);
  K(1,1) = 1 - K(1,2);
  K(2,2) = 1;
  
  for(int i =0; i < t; i++){
    K(0,0) = exp(-params(0)*lambda(1,i)/n);
    K(0,1) = 1 - K(0,0);
    
    // prediction step
    lambda_.col(i) = (lambda.col(i).t()*K).t();
    
    // update step
    
    lambda.col(i+1) = lambda_.col(i);
    lambda(1,i+1) = y(i) + (1 - params(2))*lambda_(1,i);
    
    w(i) = R::dpois(y(i), lambda_(1,i)*params(2) ,1);
  }
  double lik = accu(w);
  
  return(lik);
}
