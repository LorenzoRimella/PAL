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
      B(i,j) = xi*betas(i); 
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

void propagate_particles(arma::mat& xi,
                         vec gamma_pars,
                         int t,
                         int n_parts
){
  for(int i = 0; i < n_parts; i++){
    xi(i,t) = R::rgamma(gamma_pars(0), gamma_pars(1));
  }
}


void compute_PAL(arma::cube& q,
                 vec norm_pars,
                 arma::mat xi,
                 mat gammaprop,
                 vec gamma_pars,
                 mat& lambdabar,
                 mat prop,
                 arma::mat lambdabarresamp,
                 arma::vec params,
                 arma::mat K,
                 vec& lweights,
                 mat y,
                 int m,
                 int t,
                 int n_parts,
                 int N
){
  
  mat lambda_particles(n_parts,m);
  vec ones(9, fill::ones);
  for(int i = 0; i< n_parts; i++){
    vec mu(3, fill::zeros);
    mat transitions(9,9, fill::zeros);
    for(int k = 0; k<4; k++){
      
      update_K(K, lambdabarresamp.row(i).t(), params, xi(i,t), N,t, params(3), params(4));
      
      
      transitions = (lambdabarresamp.row(i).t()*ones.t()) %K;
      
      // if(t==0&&i==3){cout << ones*lambdabarresamp.row(i) << endl;}
      
      
      lambdabarresamp.row(i) = sum(transitions,0);
      lambdabarresamp(i,0) += 4*1025.7;
      mu(0) += transitions(0,1);
      mu(1) += transitions(3,4);
      mu(2) += transitions(6,7);
      
    }
   
    // Calculate proposale params
    double mu1 = (0.5)*((norm_pars(0) -   mu(0)*norm_pars(1)*norm_pars(1)) + sqrt((mu(0)*norm_pars(1)*norm_pars(1) - norm_pars(0))*(mu(0)*norm_pars(1)*norm_pars(1) - norm_pars(0)) + 4*y(t,0)*norm_pars(1)*norm_pars(1)));
    double mu2 = (0.5)*((norm_pars(0) -   mu(1)*norm_pars(1)*norm_pars(1)) + sqrt((mu(1)*norm_pars(1)*norm_pars(1) - norm_pars(0))*(mu(1)*norm_pars(1)*norm_pars(1) - norm_pars(0)) + 4*y(t,1)*norm_pars(1)*norm_pars(1)));
    double mu3 = (0.5)*((norm_pars(0) -   mu(2)*norm_pars(1)*norm_pars(1)) + sqrt((mu(2)*norm_pars(1)*norm_pars(1) - norm_pars(0))*(mu(2)*norm_pars(1)*norm_pars(1) - norm_pars(0)) + 4*y(t,2)*norm_pars(1)*norm_pars(1)));;
    

    
    double prop1 = sqrt(1/(y(t,0)/(mu1*mu1) + 1/(norm_pars(1)*norm_pars(1))));
    double prop2 = sqrt(1/(y(t,1)/(mu2*mu2) + 1/(norm_pars(1)*norm_pars(1))));
    double prop3 = sqrt(1/(y(t,2)/(mu3*mu3) + 1/(norm_pars(1)*norm_pars(1))));
    
    q(i,0,t) = r_truncnorm(mu1, prop1,0,1);
    q(i,1,t) = r_truncnorm(mu2, prop2,0,1);
    q(i,2,t) = r_truncnorm(mu3, prop3,0,1);
    
    // if(t==0){cout << mu(0) << endl;}
    mat transup = transitions;
    // update step
    
    transup(0,1) = (1-q(i,0,t))*transitions(0,1) + y(t,0)*q(i,0,t)*transitions(0,1)/(mu(0)*q(i,0,t));
    transup(3,4) = (1-q(i,1,t))*transitions(3,4) + y(t,1)*q(i,1,t)*transitions(3,4)/(mu(1)*q(i,1,t));
    transup(6,7) = (1-q(i,2,t))*transitions(6,7) + y(t,2)*q(i,2,t)*transitions(6,7)/(mu(2)*q(i,2,t));
    
    lambdabar.row(i) = sum(transup,0);
    lambdabar(i,0) += 4*1025.7;
    
    // lweights(i) = R::dpois(y(t,0), mu(0)*q(i,0,t) ,1) + R::dpois(y(t,1), mu(1)*q(i,1,t) ,1) + R::dpois( y(t,2),mu(2)*q(i,2,t) ,1) + R::dgamma(xi(i,t), gamma_pars(0),gamma_pars(1),1) - R::dgamma(xi(i,t), gammaprop(i,0),gammaprop(i,1),1)  + R::dbeta(q(i,0,t), beta_pars(0),beta_pars(1),1)+ R::dbeta(q(i,1,t),beta_pars(0),beta_pars(1),1)+ R::dbeta(q(i,2,t),beta_pars(0),beta_pars(1),1) - R::dbeta(q(i,0,t),betaprop(i,0),betaprop(i,1),1)- R::dbeta(q(i,1,t),betaprop(i,2),betaprop(i,3),1)- R::dbeta(q(i,2,t),betaprop(i,4),betaprop(i,5),1);
    
    // lweights(i) = R::dpois(y(t,0), mu(0)*q(i,0,t) ,1) + R::dpois(y(t,1), mu(1)*q(i,1,t) ,1) + R::dpois( y(t,2),mu(2)*q(i,2,t) ,1)  + R::dbeta(q(i,0,t), beta_pars(0),beta_pars(1),1)+ R::dbeta(q(i,1,t),beta_pars(0),beta_pars(1),1)+ R::dbeta(q(i,2,t),beta_pars(0),beta_pars(1),1) - R::dbeta(q(i,0,t),bedfgtaprop(i,0),betaprop(i,1),1)- R::dbeta(q(i,1,t),betaprop(i,2),betaprop(i,3),1)- R::dbeta(q(i,2,t),betaprop(i,4),betaprop(i,5),1);
    
    lweights(i) = R::dpois(y(t,0), mu(0)*q(i,0,t) ,1) + R::dpois(y(t,1), mu(1)*q(i,1,t) ,1) + R::dpois( y(t,2),mu(2)*q(i,2,t) ,1)+ d_truncnorm(q(i,0,t),norm_pars(0),norm_pars(1),0,1,1) + d_truncnorm(q(i,1,t),norm_pars(0),norm_pars(1),0,1,1)+ d_truncnorm(q(i,2,t),norm_pars(0),norm_pars(1),0,1,1) - d_truncnorm(q(i,0,t),mu1,prop1,0,1,1)- d_truncnorm(q(i,1,t),mu2,prop2,0,1,1) - d_truncnorm(q(i,2,t),mu3,prop3,0,1,1);  
  }
  
  
}


double computeLikelihood(vec& tWeights,
                         double lWeightsMax,
                         int n_particles){
  double tLikelihood = lWeightsMax + log(sum(exp(tWeights))) - log(n_particles);
  return tLikelihood;
}

void normaliseWeights(vec& tWeights,
                      vec& nWeights) {
  nWeights = exp(tWeights) / sum(exp(tWeights));
}


void resampleParticles(cube& q_parts,
                       cube& rq_parts,
                       mat& xi_parts,
                       mat& rxi_parts,
                       mat& rParticles,
                       mat& particles,
                       vec& weights,
                       int t,
                       int n_particles){
  
  
  
  double u = R::runif(0,1);
  int j =0;
  double cumsum_next = weights(0);
  vec indices(n_particles);
  
  for(int n=0;n<n_particles; n++){
    double Uin = (u + n)/n_particles;
    while(Uin>cumsum_next){
      j += 1;
      cumsum_next+= weights(j);
    }
    indices(n) = j;
  }
  
  
  for(int i = 0; i < n_particles ; i++) {
    rParticles.row(i) = particles.row(indices(i));
    rxi_parts(i,t) = xi_parts(indices(i),t);
    for(int k=0; k < 3; k++){
      rq_parts(i,k,t) = q_parts(indices(i),k,t);
    }
  }
}



void resampleParticlesvanilla(cube& q_parts,
                              cube& rq_parts,
                              mat& xi_parts,
                              mat& rxi_parts,
                              mat& rParticles,
                              mat& particles,
                              vec& weights,
                              int t,
                              int n_particles){
  
  // Rcpp::IntegerVector indices = Rcpp::sample(n_particles,n_particles, true, as<NumericVector>(wrap(weights)), false);
  
  double u = R::runif(0,1);
  int j =0;
  double cumsum_next = weights(0);
  vec indices(n_particles);
  
  for(int n=0;n<n_particles; n++){
    double Uin = (u + n)/n_particles;
    while(Uin>cumsum_next){
      j += 1;
      cumsum_next+= weights(j);
    }
    indices(n) = j;
  }
  
  
  
  for(int i = 0; i < n_particles ; i++) {
    rParticles.row(i) = particles.row(indices(i));
    rxi_parts(i,t) = xi_parts(indices(i),t);
    for(int k=0; k < 3; k++){
      rq_parts(i,k,t) = q_parts(indices(i),k,t);
    }
  }
}


// [[Rcpp::export(name=rotavirus_SMC_qropxi)]]
List rotavirus_SMC_qropxi( arma::vec init_dist,
                           arma::mat y_obs,
                           int m, // here m should = 9 for SIRSIRSIR
                           arma::vec regular_params,
                           vec gamma_par, 
                           vec norm_par,
                           mat prop,
                           double n_particles,
                           int ncores
){
  
  double log_lik = 0;
  int time_steps = y_obs.size()/3;
  int N = sum(init_dist);
  mat xi_particles(n_particles,time_steps, fill::zeros);
  mat xiresamp_particles(n_particles,time_steps, fill::zeros);
  mat lambdabar_particles(n_particles,m);
  mat lambdabarresamp_particles(n_particles,m);
  cube q_particles(n_particles,3,time_steps, fill::zeros);
  cube qresamp_particles(n_particles,3,time_steps, fill:: zeros);
  for(int i = 0; i< n_particles; i++){
    
    lambdabarresamp_particles.row(i) = init_dist.t(); // row or column?
    
  }
  
  
  vec log_weights(n_particles, fill::zeros);
  vec tWeights(n_particles, fill::zeros);
  vec nWeights(n_particles, fill::zeros);
  double lWeightsMax;
  
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
  mat betaprop(n_particles, 6);
  mat gammaprop(n_particles, 2);
  Rcpp::NumericVector ess(time_steps);
  // mat mu(3, n_particles);
  

  
  for(int t = 0; t< time_steps; t++){
    ess(t) = 0;

    propagate_particles(xi_particles, gamma_par, t, n_particles);
    
    
    
    
    compute_PAL(q_particles, norm_par, xi_particles, gammaprop, gamma_par, lambdabar_particles, prop, lambdabarresamp_particles, regular_params, K, log_weights, y_obs, m, t, n_particles, N);
    
    //if(t==0){cout << log_weights << endl;}
    
    lWeightsMax = max(log_weights);
    tWeights = log_weights - lWeightsMax;
    normaliseWeights(tWeights, nWeights);
    ess(t) = 1/(sum(pow( as<Rcpp::NumericVector>(wrap(nWeights)),2)));
    double ll = computeLikelihood(tWeights, lWeightsMax, n_particles);
    log_lik += ll;
    
    resampleParticlesvanilla(q_particles, qresamp_particles, xi_particles,xiresamp_particles, lambdabarresamp_particles, lambdabar_particles,nWeights,t,n_particles);
    
  }
  
  List L = List::create(Named("log_lik") = log_lik, Named("xi_particles") = xiresamp_particles, Named("q_particles") = qresamp_particles, Named("ess") = ess);
 // List L = List::create(Named("log_lik") = log_lik, Named("xi_particles") = xiresamp_particles, Named("q_particles") = qresamp_particles);
  
  return(L);
}