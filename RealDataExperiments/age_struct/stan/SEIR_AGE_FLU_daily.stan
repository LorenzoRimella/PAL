functions {
  matrix Approx_filtering(int[,] ydat, real[] params, real q[]) {
   
   vector[4] normaliser;
   vector[4] infecteds;
   real beta;
   matrix[19,4] y;
   matrix[4,4] contmat = [[params[1], params[2], params[3], params[4]], [params[2], params[5], params[6], params[7]], [params[3], params[6], params[8], params[9]], [params[4], params[7], params[9], params[10]]];
   matrix[4,4] K = [[0,0,0,0],[0,exp(-(1/1.5)),1-exp(-(1/1.5)),0],[0,0,exp(-(1/1.5)),1-exp(-(1/1.5))],[0,0,0,1]];
   matrix[4,19] out = [[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]] ;
   matrix[4,4]  filtered_intensity_pred[7*19,4];
   matrix[4,4]  filtered_intensity_update[7*19,4];
   matrix[4,4]  state[134];
   
   
    state[1,1] = [949-1,0,1,0]; // 949
    state[1,2] = [1690-1,0,1,0]; // 1690
    state[1,3] = [3467-1,0,1,0]; // 3467
    state[1,4] = [1894-1,0,1,0]; // 1894
    for(t in 1:19){
      normaliser = [0,0,0,0]';
        for(j in 1:7){
        infecteds = col(state[7*(t-1) + j],3);
          for(k in 1:4){
          // get force of infection
          beta = dot_product(infecteds,row(contmat,k));

          K[1,1] = exp(-0.1428571*beta/8000);
          K[1,2] = 1-K[1,1];

          filtered_intensity_pred[7*(t-1)+j,k] = (((row(state[7*(t-1)+j],k)')*[1,1,1,1])).*K;

          state[7*(t-1)+j+1,k] = [1,1,1,1]*filtered_intensity_pred[7*(t-1)+j,k];
          normaliser[k] = normaliser[k] + filtered_intensity_pred[7*(t-1)+j,k,2,3];
          }
        }
        
        for(k in 1:4){
          out[k,t] = normaliser[k];
        }
        
          for(k in 1:4){
          // update step
          // unobserved states remain untouched
          filtered_intensity_update[7*(t),k] = filtered_intensity_pred[7*(t),k];
          // update the observed E -> I
          
          filtered_intensity_update[7*t,k,2,3] = (filtered_intensity_pred[7*(t),k,2,3]/normaliser[k])*ydat[t,k] + (1-q[k])*filtered_intensity_pred[7*(t),k,2,3];
          if(t<19){state[7*(t)+1,k] = [1,1,1,1]*filtered_intensity_update[7*(t),k];}
          }
        }
        
  return(out);
  }
}


// The input data is a vector 'y' of length 'N'.
data {
  int y[19,4];
}

// The parameters accepted by the model
parameters {
  real<lower = 0> params[10]; // Model parameters
  real<lower = 0, upper = 1> q[4];
}


// generate approx filtered intensity vectors
transformed parameters{
  matrix[4,19] filtered_rate;
  filtered_rate = Approx_filtering(y, params, q);
}

// The model to be estimated.
model {
    params ~ gamma(5, 1);
    q    ~ normal(0.5, 0.5);
    for(t in 1:19){
      for(k in 1:4){
    y[t,k] ~ poisson(filtered_rate[k,t]*q[k]);
    
}
}
}


generated quantities {
  real log_lik = 0;
  for(t in 1:19){
      for(k in 1:4){
  log_lik += poisson_lpmf(y[t,k] | filtered_rate[k,t]*q[k]);
    }
  }
}
