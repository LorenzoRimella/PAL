// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <math.h>
using namespace Rcpp;
using namespace arma;


// params is of the form (beta, rho, gamma,h)
// [[Rcpp::export(name=SEIRsim)]]
mat SEIRsim(arma::mat init_pop,
            arma::vec params,
            double q,
            double h){
  
  // define initial conditions
  int T = 19;
  int n = arma::accu(init_pop); // total pop
  cube state(4,4,T);
  mat contmat = {{params(0),params(1),params(2),params(3)},{params(1),params(4),params(5),params(6)},{params(2),params(5),params(7),params(8)},{params(3),params(6),params(8),params(9)}};
  mat obs(4,T);
  double beta;
  double new_exp;
  double new_rem;
  mat new_inf(4,T);
  rowvec infecteds = init_pop.row(2);
  
  state.slice(0) = init_pop.t();
  
  for(int i = 0; i < T-1; i++) {
    
    for(int k = 0; k<4; k++){
      
      // calculate force of infection
      beta = dot(infecteds,contmat.col(k))/n;
      
      
      // simulate new movements between compartments
      new_exp = R::rbinom(state(k,0,i),1-exp(-h*(1/7)*beta));
      new_inf(k,i) = R::rbinom(state(k,1,i),1-exp(-1/1.5));
      new_rem = R::rbinom(state(k,2,i),1-exp(-1/1.5));
      
      // update states at next timestep
      state(k,0,i+1) = state(k,0,i) - new_exp;
      state(k,1,i+1) = state(k,1,i) + new_exp - new_inf(k,i);
      state(k,2,i+1) = state(k,2,i) + new_inf(k,i) - new_rem;
      state(k,3,i+1) = state(k,3,i) + new_rem;
      
      
      
    }
    for(int k = 0; k<4; k++){
      infecteds(k) = state(k,2,i+1);
    }
  }
  
  for(int t = 0; t<T; t++)
    for(int j = 0; j < 4 ; j++){
      obs(j,t) = R::rbinom(new_inf(j,t), q);
    }
    
    
    
    return(obs);
}


// [[Rcpp::export(name=SEIR_daily_sim)]]
mat SEIR_daily_sim(arma::mat init_pop,
                   arma::vec params,
                   double q,
                   double h){
  
  // define initial conditions
  int T = 19;
  int n = arma::accu(init_pop); // total pop
  cube state(4,4,7*T);
  mat contmat = {{params(0),params(1),params(2),params(3)},{params(1),params(4),params(5),params(6)},{params(2),params(5),params(7),params(8)},{params(3),params(6),params(8),params(9)}};
  mat obs(4,T);
  double beta;
  double new_exp;
  double new_rem;
  double new_inf;
  mat new_weekly_inf(4,7*T, fill::zeros);
  rowvec infecteds = init_pop.row(2);
  
  state.slice(0) = init_pop.t();
  
  for(int i = 0; i < T-1; i++) {
    
    for(int j = 0; j< 7; j++){
      for(int k = 0; k<4; k++){
        
        // calculate force of infection
        beta = dot(infecteds,contmat.col(k))/n;
        
        
        // simulate new movements between compartments
        new_exp = R::rbinom(state(k,0,i*7 + j),1-exp(-(0.1428571)*beta));
        new_inf = R::rbinom(state(k,1,i*7 + j),1-exp(-1/1.5));
        new_rem = R::rbinom(state(k,2,i*7 + j),1-exp(-1/1.5));
        
        // update states at next timestep
        state(k,0,i*7 + j+1) = state(k,0,i*7 + j) - new_exp;
        state(k,1,i*7 + j+1) = state(k,1,i*7 + j) + new_exp - new_inf;
        state(k,2,i*7 + j+1) = state(k,2,i*7 + j) + new_inf - new_rem;
        state(k,3,i*7 + j+1) = state(k,3,i*7 + j) + new_rem;
        
        
        new_weekly_inf(k,i) += new_inf;
      }
      for(int k = 0; k<4; k++){
        infecteds(k) = state(k,2,i*7 + j+1);
      }
    }
  }
  
  for(int t = 0; t<T; t++)
    for(int j = 0; j < 4 ; j++){
      obs(j,t) = R::rbinom(new_weekly_inf(j,t), q);
    }
    
    
    
    return(obs);
}



// [[Rcpp::export(name=SEIR_daily_sim_det)]]
mat SEIR_daily_sim_det(arma::mat init_pop,
                       arma::vec params,
                       double q,
                       double h){
  
  // define initial conditions
  int T = 19;
  int n = arma::accu(init_pop); // total pop
  cube state(4,4,7*T);
  mat contmat = {{params(0),params(1),params(2),params(3)},{params(1),params(4),params(5),params(6)},{params(2),params(5),params(7),params(8)},{params(3),params(6),params(8),params(9)}};
  mat obs(4,T);
  double beta;
  double new_exp;
  double new_rem;
  double new_inf;
  mat new_weekly_inf(4,7*T, fill::zeros);
  rowvec infecteds = init_pop.row(2);
  
  state.slice(0) = init_pop.t();
  
  for(int i = 0; i < T-1; i++) {
    
    for(int j = 0; j< 7; j++){
      for(int k = 0; k<4; k++){
        
        // calculate force of infection
        beta = dot(infecteds,contmat.col(k))/n;
        
        
        // simulate new movements between compartments
        new_exp = (state(k,0,i*7 + j))*(1-exp(-h*beta));
        new_inf = (state(k,1,i*7 + j))*(1-exp(-h/1.5));
        new_rem = (state(k,2,i*7 + j))*(1-exp(-h/1.5));
        
        // update states at next timestep
        state(k,0,i*7 + j+1) = state(k,0,i*7 + j) - new_exp;
        state(k,1,i*7 + j+1) = state(k,1,i*7 + j) + new_exp - new_inf;
        state(k,2,i*7 + j+1) = state(k,2,i*7 + j) + new_inf - new_rem;
        state(k,3,i*7 + j+1) = state(k,3,i*7 + j) + new_rem;
        
        
        new_weekly_inf(k,i) += new_inf;
      }
      for(int k = 0; k<4; k++){
        infecteds(k) = state(k,2,i*7 + j+1);
      }
    }
  }
  
  for(int t = 0; t<T; t++)
    for(int j = 0; j < 4 ; j++){
      obs(j,t) = R::rpois(new_weekly_inf(j,t)*q);
    }
    
    return(obs);
}

// [[Rcpp::export(name=SEIR_daily_sim_pred)]]
mat SEIR_daily_sim_pred(arma::mat init_pop,
                        arma::vec params,
                        double q,
                        double h,
                        int T){
  
  // define initial conditions
  int n = arma::accu(init_pop); // total pop
  cube state(4,4,7*T);
  mat contmat = {{params(0),params(1),params(2),params(3)},{params(1),params(4),params(5),params(6)},{params(2),params(5),params(7),params(8)},{params(3),params(6),params(8),params(9)}};
  mat obs(4,T);
  double beta;
  double new_exp;
  double new_rem;
  double new_inf;
  mat new_weekly_inf(4,T, fill::zeros);
  rowvec infecteds = init_pop.row(2);
  
  state.slice(0) = init_pop.t();
  
  for(int i = 0; i < T-1; i++) {
    
    for(int j = 0; j< 7; j++){
      for(int k = 0; k<4; k++){
        
        // calculate force of infection
        beta = dot(infecteds,contmat.col(k))/n;
        
        
        // simulate new movements between compartments
        new_exp = R::rbinom(state(k,0,i*7 + j),1-exp(-h*beta));
        new_inf = R::rbinom(state(k,1,i*7 + j),1-exp(-h/1.5));
        new_rem = R::rbinom(state(k,2,i*7 + j),1-exp(-h/1.5));
        
        // update states at next timestep
        state(k,0,i*7 + j+1) = state(k,0,i*7 + j) - new_exp;
        state(k,1,i*7 + j+1) = state(k,1,i*7 + j) + new_exp - new_inf;
        state(k,2,i*7 + j+1) = state(k,2,i*7 + j) + new_inf - new_rem;
        state(k,3,i*7 + j+1) = state(k,3,i*7 + j) + new_rem;
        
        
        new_weekly_inf(k,i) += new_inf;
      }
      for(int k = 0; k<4; k++){
        infecteds(k) = state(k,2,i*7 + j+1);
      }
    }
  }
  
  for(int t = 0; t<T; t++)
    for(int j = 0; j < 4 ; j++){
      cout << new_weekly_inf(j,t) << endl;
      obs(j,t) = R::rbinom(new_weekly_inf(j,t), q);
    }
    
    
    
    return(obs);
}


// [[Rcpp::export(name=SEIR_det_pred) ]]
mat SEIR_det_pred(arma::mat init_pop,
                  arma::vec params,
                  double q,
                  double h,
                  int T){
  
  // define initial conditions
  int n = arma::accu(init_pop); // total pop
  cube state(4,4,7*T);
  mat contmat = {{params(0),params(1),params(2),params(3)},{params(1),params(4),params(5),params(6)},{params(2),params(5),params(7),params(8)},{params(3),params(6),params(8),params(9)}};
  mat obs(4,T);
  double beta;
  double new_exp;
  double new_rem;
  double new_inf;
  mat new_weekly_inf(4,7*T, fill::zeros);
  rowvec infecteds = init_pop.row(2);
  
  state.slice(0) = init_pop.t();
  
  for(int i = 0; i < T-1; i++) {
    
    for(int j = 0; j< 7; j++){
      for(int k = 0; k<4; k++){
        
        // calculate force of infection
        beta = dot(infecteds,contmat.col(k))/n;
        
        
        // simulate new movements between compartments
        new_exp = (state(k,0,i*7 + j))*(1-exp(-h*beta));
        new_inf = (state(k,1,i*7 + j))*(1-exp(-h/1.5));
        new_rem = (state(k,2,i*7 + j))*(1-exp(-h/1.5));
        
        // update states at next timestep
        state(k,0,i*7 + j+1) = state(k,0,i*7 + j) - new_exp;
        state(k,1,i*7 + j+1) = state(k,1,i*7 + j) + new_exp - new_inf;
        state(k,2,i*7 + j+1) = state(k,2,i*7 + j) + new_inf - new_rem;
        state(k,3,i*7 + j+1) = state(k,3,i*7 + j) + new_rem;
        
        
        new_weekly_inf(k,i) += new_inf;
      }
      for(int k = 0; k<4; k++){
        infecteds(k) = state(k,2,i*7 + j+1);
      }
    }
  }
  
  for(int t = 0; t<T; t++)
    for(int j = 0; j < 4 ; j++){
      obs(j,t) = R::rpois(new_weekly_inf(j,t)*q);
    }
    
    return(obs);
}

// [[Rcpp::export(name=SEIR_det_coarse) ]]
mat SEIR_det_coarse(arma::mat init_pop,
                    arma::vec params,
                    double q,
                    double h,
                    int T){
  
  // define initial conditions
  int n = arma::accu(init_pop); // total pop
  cube state(4,4,14*T);
  mat contmat = {{params(0),params(1),params(2),params(3)},{params(1),params(4),params(5),params(6)},{params(2),params(5),params(7),params(8)},{params(3),params(6),params(8),params(9)}};
  mat obs(4,T);
  double beta;
  double new_exp;
  double new_rem;
  double new_inf;
  mat new_weekly_inf(4,14*T, fill::zeros);
  rowvec infecteds = init_pop.row(2);
  
  state.slice(0) = init_pop.t();
  
  for(int i = 0; i < T-1; i++) {
    
    for(int j = 0; j< 14; j++){
      for(int k = 0; k<4; k++){
        
        // calculate force of infection
        beta = dot(infecteds,contmat.col(k))/n;
        
        
        // simulate new movements between compartments
        new_exp = (state(k,0,i*14 + j))*(1-exp(-h*beta));
        new_inf = (state(k,1,i*14 + j))*(1-exp(-h/3));
        new_rem = (state(k,2,i*14 + j))*(1-exp(-h/3));
        
        // update states at next timestep
        state(k,0,i*14 + j+1) = state(k,0,i*14 + j) - new_exp;
        state(k,1,i*14 + j+1) = state(k,1,i*14 + j) + new_exp - new_inf;
        state(k,2,i*14 + j+1) = state(k,2,i*14 + j) + new_inf - new_rem;
        state(k,3,i*14 + j+1) = state(k,3,i*14 + j) + new_rem;
        
        
        new_weekly_inf(k,i) += new_inf;
      }
      for(int k = 0; k<4; k++){
        infecteds(k) = state(k,2,i*14 + j+1);
      }
    }
  }
  
  for(int t = 0; t<T; t++)
    for(int j = 0; j < 4 ; j++){
      obs(j,t) = R::rpois(new_weekly_inf(j,t)*q);
    }
    
    return(obs);
}

