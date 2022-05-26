// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <math.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(BH)]]
// [[Rcpp::plugins(openmp)]]


mat prediction_step(double beta,
                    mat& K,
                    vec lambda,
                    vec ones,
                    int h
){
  K(0,0) = exp(-beta);
  K(0,1) =  1 - exp(-beta);
  mat out = (lambda*ones.t())%(K);
  return(out);
}

mat update_step(mat Lambda_,
                mat Norm,
                mat Q,
                mat Y,
                vec ones
){
  
  mat Normnan = (Lambda_/Norm);
  // cout << Norm<< endl;
  mat out = Y%Normnan + (ones*ones.t() - Q)%Lambda_;
  
  return(out.transform( [](double val) { return (std::isnan(val) ? double(0) : val); }));
}




void calculate_beta(vec& betas,
                    mat v,
                    vec n,
                    double beta,
                    double gg,
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
        sum_term += (gg*v(i,l)/n(i))*( (ilambda(l)/n(l)) - (ilambda(i)/n(i)) );}
    }
    betas(i) = beta*sum_term;
  }
}


double calculate_likelihood(mat A,
                            mat Y
){
  
  
  double out = 0;
  
  for(int i = 0; i< Y.n_rows; i++){
    for(int j = 0; j< Y.n_rows; j++){
      if(A(i,j) ==0 ){out += 0;}
      else{
        out += -A(i,j) + Y(i,j)*(log(A(i,j)));
      }
    }
  }
  
  return(out);
  
}








// param is (beta, rho, gamma)
// [[Rcpp::export(name=PoiGravity)]]
double  PoiGravity(arma::vec n,
                 arma::mat y,
                 arma::mat init_dist,
                 arma::mat vg,
                 arma::mat births,
                 arma::vec params,
                 arma::vec q,
                 arma::vec survivalprob,
                 double c,
                 double g,
                 double h,
                 double a,
                 int m,
                 int k,
                 int ncores,
                 int tstar
){
  
  // initialise length of series
  int T = y.n_rows;
  
  // initialise filtered intensity matrices for each city
  cube  Lambda_(m,m,4*40, fill::zeros);
  cube  Lambda(m,m,40, fill::zeros);
  cube  Norms(m,m,40, fill::zeros);
  //
  
  
  // initialise intensity vectors
  arma::cube lambda(m,k,T*4+1, fill::zeros);
  // initialise initial distributions
  
  lambda.slice(0) = init_dist;
  
  // build useful ones vector
  vec ones(m,fill::ones);
  
  // multiply v by g
  mat v = vg;
  // Initialise beta vector
  vec betas(k, fill::zeros);
  // Initialise likelihood terms
  mat w(k,T, fill::zeros);

  
  // time varying beta
  double b = (1 - 2*0.739*a)*params[0];
  // begin filtering
  for(int t=0; t<T; t++){
 
    // update beta
    if((t)%26 == 1 || (t)%26 == 8 || (t)%26 == 18 || (t)%26 == 22 ){ b = (1 + ((1-0.739)/0.739)*a)*params[0];}
    if((t)%26 == 7 || (t)%26 == 14 || (t)%26 == 21 || (t)%26 == 25 ){ b = (1 - a)*params[0];}
    
    
    
    for(int s = 0; s<4; s++){
      calculate_beta(betas, v, n, b, g, lambda.slice(t*4+s).row(2).t(), k, ones);
      
     
      for(int i=0; i<k; i++){
        
        // initialise K transition matrix
        mat K(m,m,fill::zeros);
        
        for(int j=1; j<params.size() ;j++){
          K(j,j) = exp(-h*params[j]);
          K(j,j+1) =  1 - exp(-h*params[j]);
        }
        K(3,3) = 1;

        // Prediction step
        Lambda_.slice(4*i+s) = prediction_step(betas(i), K, lambda.slice(t*4+s).col(i),ones, h);
        
        
        lambda.slice(t*4+s+1).col(i) = sum(Lambda_.slice(4*i+s),0).t();

        lambda(0,i,t*4+s+1)  += ((1-c)/(26*4))*births(t,i);

      }
    }
    
    
    for(int i=0; i<k; i++){
      
      mat Norm = Lambda_.slice(4*i)+Lambda_.slice(4*i+1)+Lambda_.slice(4*i+2)+Lambda_.slice(4*i+3);
      
      // build Q and obs mat
      mat Q(m,m, fill::zeros);
      Q(2,3) = q(i);
      mat Y(m,m, fill::zeros);
      Y(2,3) = y(t,i);
      
      // create matrix for likelihood computation
      mat LQ = Norm%Q;
      // update step
      Lambda.slice(i) = update_step(Lambda_.slice(4*i+3), Norm ,Q,Y, ones);
      
      
      // update the intensity vectors.
      // take deaths
      if(t != T-1){lambda.slice(t*4+4).col(i) = (sum(Lambda.slice(i),0).t())%survivalprob ;
        // add cohort fraction of births on 36th week of year - start of school year.
        if(t%26 == 18){lambda(0,i,4*t+4) += c*births(t,i);
        }
        // add the rest
        lambda(0,i,t*4+4)  += ((1-c)/(26*4))*births(t,i);
      }
      // calculate approx log-likelihood :)
      if(t>tstar){
        w(i,t) = calculate_likelihood(LQ, Y);
      }
    }
  }
  
  double lik = accu(w);
  
  // List out = List::create(Named("Lambda_") = Lambda_, Named("Lambda") = Lambda, Named("lambda") = lambda, Named("lik") = lik, Named("w") = w, Named("Norms")=Norms);
  
  //List out = List::create(Named("lambda") = lambda);
  return(lik);
}

