// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <math.h>
using namespace Rcpp;
using namespace arma;

IntegerVector rmultinom_1(unsigned int size, NumericVector probs, unsigned int N) {
  IntegerVector outcome(N);
  rmultinom(size, probs.begin(), N, outcome.begin());
  return outcome;
}


mat rmultinom_rcpp(unsigned int n, unsigned int size, NumericVector probs) {
  unsigned int N = probs.length();
  IntegerMatrix sim(N, n);
  for (unsigned int i = 0; i < n; i++) {
    sim(_,i) = rmultinom_1(size, probs, N);
  }
  mat out = as<arma::mat>(sim);
  return out;
}


void calculate_beta(vec& betas,
                    mat v,
                    vec n,
                    double beta,
                    vec ilambda,
                    int k,
                    vec ones
){
  
  // calculate gravity model adjusted infection rate
  
  for(int i = 0; i<k; i++){
    // initialise as normal
    double sum_term = (ilambda(i)/n(i));
    
    // add interaction terms
    for(int l = 0; l<k; l++){
      if(l != i){
        sum_term += (v(i,l)/n(i))*( (ilambda(l)/n(l)) - (ilambda(i)/n(i)) );}
    }
    betas(i) = beta*sum_term;
  }
}

// [[Rcpp::export]]
mat Sim_Gravity(arma::vec n,
                 arma::mat init_dist,
                 arma::mat vg,
                 arma::mat births,
                 arma::vec params,
                 arma::vec q,
                 arma::vec survivalprob,
                 double T,
                 int n_steps,
                 double c,
                 double g,
                 double h,
                 double a,
                 int m,
                 int K){
  
  // build useful ones vector
  vec ones(m,fill::ones);
  // initialise observations objects
  mat transitionsItoR(K,T*n_steps, fill::zeros);
  mat cumulative_transitions(K,T, fill::zeros);
  mat observations(K,T);
  
  int new_exp = 0;
  int new_inf = 0;
  int new_rem = 0;
  // sort out preliminaries with both observation models
  int total_pop = sum(n);
  

  // initialize states for both observation models
  cube state(4,K,T*n_steps +1 , fill::zeros);
  

  // draw initial distribution
  for(int i =0; i < K; i++){
    //  draw initial distribution
    NumericVector prob = as<Rcpp::NumericVector>(wrap(init_dist.col(i)));
    state.slice(0).col(i) = rmultinom_rcpp(1,n(i), prob);
  }
  
  // multiply v by g
  mat v = g*vg;
  // initialise gravity infection rates
  mat outcheck( K,T*4 ,fill::ones);
  vec betas(K, fill::zeros);
  
  double b = (1 - 2*0.739*a)*params[0];
  
  for(int t = 0; t< T; t++){
    // time varying beta
   // if((t)%26 == 1 || (t)%26 == 8 || (t)%26 == 18 || (t)%26 == 22 ){ b = (1 + 2*(1-0.739)*a)*params[0];}
   // if((t)%26 == 7 || (t)%26 == 14 || (t)%26 == 21 || (t)%26 == 25 ){ b = (1 - 2*0.739*a)*params[0];}
    if((t)%26 == 1 || (t)%26 == 8 || (t)%26 == 18 || (t)%26 == 22 ){ b = (1 + ((1-0.739)/0.739)*a)*params[0];}
    if((t)%26 == 7 || (t)%26 == 14 || (t)%26 == 21 || (t)%26 == 25 ){ b = (1 - a)*params[0];}
    // deaths
    
    if(t>0){
    for(int k =0; k < K; k++){
      state(0, k, t*n_steps) = R::rbinom(state(0,k,t*n_steps ),survivalprob(0));
      state(1, k, t*n_steps) = R::rbinom(state(1,k,t*n_steps ),survivalprob(1));
      state(2, k, t*n_steps) = R::rbinom(state(2,k,t*n_steps ),survivalprob(2));
      state(3, k, t*n_steps) = R::rbinom(state(3,k,t*n_steps ),survivalprob(3));
    }
    }
    
    
    for(int i = 0; i< n_steps; i++){
      
      calculate_beta( betas, v, n, b, state.slice(t*n_steps + i).row(2).t(), K, ones);
      
      for(int k =0; k < K; k++){
        
        // sample transitions
        new_exp = R::rbinom(state(0,k,t*n_steps + i),1-exp(-h*betas(k)));
        outcheck(k,t+n_steps) = 1-exp(-h*betas(k));
        new_inf = R::rbinom(state(1,k,t*n_steps + i),1-exp(-h*params[1]));

        new_rem = R::rbinom(state(2,k,t*n_steps + i),1-exp(-h*params[2]));

        // add to cumulative count
        cumulative_transitions(k,t) += new_rem;
        
        // calculate new states
        
        state(0, k, t*n_steps + i + 1) = state(0,k,t*n_steps + i) - new_exp;
        state(1, k, t*n_steps + i + 1) = state(1,k,t*n_steps + i) + new_exp - new_inf;
        state(2, k, t*n_steps + i + 1) = state(2,k,t*n_steps + i) + new_inf - new_rem;
        state(3, k, t*n_steps + i + 1) = state(3,k,t*n_steps + i) + new_rem;
    }
      for(int k =0; k < K; k++){
        state(0, k, t*n_steps + i + 1) += R::rpois(((1-c)/(26*n_steps))*births(t,k));
      }
      
  }
   
   // add cohort births
   if(t%26 == 18){
     for(int k =0; k < K; k++){
       state(0, k, (t+1)*(n_steps)) += R::rpois(c*births(t,k));
     }
   }
}
  for(int t = 0; t < T; t++){
    for(int k = 0; k<K; k++){
      observations(k,t) =  R::rbinom(cumulative_transitions(k,t),q(k));
    }
  }
  
  
  return(observations);
}
  
  
  