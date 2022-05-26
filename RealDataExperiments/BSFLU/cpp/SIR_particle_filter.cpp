// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <math.h>
using namespace Rcpp;
using namespace arma;

void propagate_particles(mat resamp_parts,
                         mat& parts,
                         vec pars,
                         int n_parts,
                         int pop){
  int new_inf;
  int new_rem;
  
  for(int j = 0; j< n_parts; j++){
    new_inf = R::rbinom(resamp_parts(0,j), 1 - exp(-pars(0)*resamp_parts(1,j)/pop));
    new_rem = R::rbinom(resamp_parts(1,j), 1 - exp(-pars(1)));
    
    parts(0,j) = resamp_parts(0,j) - new_inf;
    parts(1,j) = resamp_parts(1,j) + new_inf - new_rem;
    parts(2,j) = resamp_parts(2,j) + new_rem;
     
  }
}


void weight_particles( vec& weights,
                       double obs,
                       int n_parts,
                       vec pars,
                       mat parts){
  
  for(int i = 0; i< n_parts; i++){
    // weights(i) = R::dbinom(obs, parts(1,i), pars(2), 1);
       weights(i) = R::dpois(obs, parts(1,i)*pars(2), 1);
  }
}

void normaliseWeights(vec& tWeights,
                      vec& nWeights) {
  nWeights = exp(tWeights) / sum(exp(tWeights));
}

double computeLikelihood(vec& tWeights,
                         double lWeightsMax,
                         int n_particles){
  double tLikelihood = lWeightsMax + log(sum(exp(tWeights))) - log(n_particles);
  return tLikelihood;
}

void resampleParticles(mat& rParticles,
                       mat& particles,
                       vec& weights,
                       int n_particles){
  Rcpp::IntegerVector indices = Rcpp::sample(n_particles,n_particles, true, as<NumericVector>(wrap(weights)), false);
  for(int i = 0; i < n_particles ; i++) {
    rParticles.col(i) = particles.col(indices(i));
  }
}



// [[Rcpp::export(name=Particle_likelihood_SIR)]]
double Particle_likelihood_SIR(vec init_pop,
                                 vec y,
                                 arma::vec params,
                                 int n_particles){

  double logLikelihood = 0;
  int n = accu(init_pop);
  int t = y.size();
  mat particles(3,n_particles);
  mat resampled_particles(3,n_particles);
  for(int i = 0; i<n_particles; i++){
  resampled_particles.col(i) = init_pop;
  }
  vec  weights(n_particles);
  vec tWeights;
  vec nWeights(n_particles);
  double lWeightsMax;
  
  
  for(int i = 0; i < t; i++){
    propagate_particles(resampled_particles, particles, params, n_particles, n);
    weight_particles(weights, y(i), n_particles, params, particles);

    lWeightsMax = max(weights);
    
    tWeights = weights - lWeightsMax;
    
    normaliseWeights(tWeights, nWeights);
    double ll = computeLikelihood(tWeights, lWeightsMax, n_particles);
    logLikelihood += ll;
    
    resampleParticles(resampled_particles, particles, nWeights,n_particles);
  }
  return(logLikelihood);
}