// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <math.h>
#define RCPPDIST_DONT_USE_ARMA
#include <RcppDist.h>
// [[Rcpp::depends(RcppDist)]]
using namespace Rcpp;
using namespace arma;


void update_K(mat& K,
              vec lambar,
              vec betas,
              double xi,
              int N,
              int t,
              double phase,
              double rho
){
  // seasonal component
  
  mat B(3,3, fill::zeros);
  for(int i = 0; i<3;i++){
    for(int j = 0; j<3;j++){
      B(i,j) = betas(i); 
    }
  }
  //if(t==0){cout << B(0,0) << endl;}
  
  double kappa = (1 + rho*cos(2*3.141593*t/52 + phase));
  
  vec I;
  I = {lambar(1),lambar(4),lambar(7)};
  vec lambda = B*I*kappa/N;
  K(0,1) = 1-exp(-0.25*lambda(0));
  K(3,4) = 1-exp(-0.25*lambda(1));
  K(6,7) = 1-exp(-0.25*lambda(2));
  
  
  K(0,0)= 1 - K(0,1)-  K(0,2) - K(0,3) - K(0,4) - K(0,5) - K(0,6) - K(0,7) - K(0,8);
  K(1,1)= 1 - K(1,0) - K(1,2) - K(1,3) - K(1,4) - K(1,5) - K(1,6) - K(1,7) - K(1,8);
  K(2,2)= 1 - K(2,0) - K(2,1) - K(2,3) - K(2,4) - K(2,5) - K(2,6) - K(2,7) - K(2,8);
  K(3,3)= 1 - K(3,0) - K(3,1) - K(3,2) - K(3,4) - K(3,5) - K(3,6) - K(3,7) - K(3,8);
  K(4,4)= 1 - K(4,0) - K(4,1) - K(4,2) - K(4,3) - K(4,5) - K(4,6) - K(4,7) - K(4,8);
  K(5,5)= 1 - K(5,0) - K(5,1) - K(5,2) - K(5,3) - K(5,4) - K(5,6) - K(5,7) - K(5,8);
  K(6,6)= 1 - K(6,0) - K(6,1) - K(6,2) - K(6,3) - K(6,4) - K(6,5) - K(6,7) - K(6,8);
  K(7,7)= 1 - K(7,0) - K(7,1) - K(7,2) - K(7,3) - K(7,4) - K(7,5) - K(7,6) - K(7,8);
  K(8,8)= 1 - K(8,0) - K(8,1) - K(8,2) - K(8,3) - K(8,4) - K(8,5) - K(8,6) - K(8,7);
  // if(t==0){cout << K(0,1) << endl;}
}




// [[Rcpp::export(name=rotavirus_equidispersed)]]
List rotavirus_equidispersed( arma::vec init_dist,
                           arma::mat y_obs,
                           int m, // here m should = 9 for SIRSIRSIR
                           arma::vec regular_params,
                           double q
){
  double log_lik = 0;
  int time_steps = y_obs.size()/3;
  int N = 82372825;
  mat lambda(time_steps*4, 9, fill::zeros);
  lambda.row(0) = init_dist.t();
  
  cube Lambda_(9,9,4,fill::zeros);
  mat weekly_incidence(9,9, fill::zeros);
  mat K(9,9,fill::value(0));
  double h = 0.25;
  // Recovery rates;
  double gamma = 1;
  K(1,2) = 1 - exp(-h*gamma);
  K(4,5) = 1 - exp(-h*gamma);
  K(7,8) = 1 - exp(-h*gamma);
  // aging rates
  double delta_1 = 0.003636364;
  double delta_2 = 0.0003496503;
  K(0,3) = 1 - exp(-h*delta_1);
  K(1,4) = 1 - exp(-h*delta_1);
  K(2,5) = 1 - exp(-h*delta_1);
  K(3,6) = 1 - exp(-h*delta_2);
  K(4,7) = 1 - exp(-h*delta_2);
  K(5,8) = 1 - exp(-h*delta_2);
  // immunity waning rates;
  double omega =  0.01923077;
  K(2,0) = 1 - exp(-h*omega);
  K(5,3) = 1 - exp(-h*omega);
  K(8,6) = 1 - exp(-h*omega);
  
  vec ones(9, fill::ones);
  
  for(int t = 1; t<time_steps; t++){
    for(int i = 0; i<4; i++){
      
      update_K(K,lambda.row(4*(t)-4+i).t(), regular_params,1, N, t, regular_params(3), regular_params(4));
      
      Lambda_.slice(i)  =  (lambda.row(4*(t)-4+i).t()*ones.t()) %K;
      lambda.row(4*(t)-4+i+1) = sum(Lambda_.slice(i),0);
      lambda(4*(t)-4+i+1,0) += 4*1025.7;
    }
    
    mat Norm = Lambda_.slice(0) + Lambda_.slice(1) + Lambda_.slice(2) + Lambda_.slice(3);
    mat Lambda = Lambda_.slice(3);
    
    Lambda(0,1) = (1-q)*Lambda_(0,1,3) + y_obs(t-1,0)*Lambda_(0,1,3)/Norm(0,1); 
    Lambda(3,4) = (1-q)*Lambda_(3,4,3)+ y_obs(t-1,1)*Lambda_(3,4,3)/Norm(3,4) ; 
    Lambda(6,7) = (1-q)*Lambda_(6,7,3) + y_obs(t-1,2)*Lambda_(6,7,3)/Norm(6,7); 
    // cout << t <<endl;
    // log_lik += R::dpois(y_obs(t-1,0), Norm(0,1)*mu1, 1) + R::dpois(y_obs(t-1,1), Norm(3,4)*mu2, 1)  + R::dpois(y_obs(t-1,2), Norm(6,7)*mu3, 1) + d_truncnorm(mu1, norm_par(0), norm_par(1),0,1,1) +d_truncnorm(mu2, norm_par(0), norm_par(1),0,1,1) +d_truncnorm(mu3, norm_par(0), norm_par(1),0,1,1)-d_truncnorm(mu1,mu1,prop1,0,1,1) -d_truncnorm(mu1,mu1,prop2,0,1,1) -d_truncnorm(mu1,mu1,prop3,0,1,1)  ;
    
    log_lik += R::dpois(y_obs(t-1,0), Norm(0,1)*q, 1) + R::dpois(y_obs(t-1,1), Norm(3,4)*q, 1)  + R::dpois(y_obs(t-1,2) , Norm(6,7)*q, 1);
      
    lambda.row(4*(t)) = sum(Lambda,0);
    lambda(4*(t),0) += 4*1025.7;
  }
  List out = List::create(Named("log_lik") = log_lik);
  
  return(out);
}
