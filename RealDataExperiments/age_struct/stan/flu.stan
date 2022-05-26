functions {
  vector SEIR_matrix_sym(real time, vector y, real[] params) {
    vector[20] dydt;
    real IR1;
    real InR1;
    real RR1;
    real InR2;
    real RR2;
    real InR3;
    real RR3;
    real InR4;
    real RR4;
    real k21;
    real k31;
    real k32;
    real k41;
    real k42;
    real k43;
    real RI1;
    real RI2;
    real RI3;
    real RI4;
    real IR2;
    real IR3;
    real IR4;
    IR1 = y[1]*(params[1]*y[3]+params[2]*y[7]+params[3]*y[11]+params[4]*y[15])/8000;
    InR1 = y[2]/0.2142857143;
    RR1 = y[3]/0.2142857143;
    InR2 = y[6]/0.2142857143;
    RR2 = y[7]/0.2142857143;
    InR3 = y[10]/0.2142857143;
    RR3 = y[11]/0.2142857143;
    InR4 = y[14]/0.2142857143;
    RR4 = y[15]/0.2142857143;
    k21 = params[2];
    k31 = params[3];
    k32 = params[6];
    k41 = params[4];
    k42 = params[7];
    k43 = params[9];
    RI1 = InR1;
    RI2 = InR2;
    RI3 = InR3;
    RI4 = InR4;
    IR2 = y[5]*(k21*y[3]+params[5]*y[7]+params[6]*y[11]+params[7]*y[15])/8000;
    IR3 = y[9]*(k31*y[3]+k32*y[7]+params[8]*y[11]+params[9]*y[15])/8000;
    IR4 = y[13]*(k41*y[3]+k42*y[7]+k43*y[11]+params[10]*y[15])/8000;
    dydt[1] = -IR1;
    dydt[2] = IR1-InR1;
    dydt[3] = InR1-RR1;
    dydt[4] = RR1;
    dydt[5] = -IR2;
    dydt[6] = IR2-InR2;
    dydt[7] = InR2-RR2;
    dydt[8] = RR2;
    dydt[9] = -IR3;
    dydt[10] = IR3-InR3;
    dydt[11] = InR3-RR3;
    dydt[12] = RR3;
    dydt[13] = -IR4;
    dydt[14] = IR4-InR4;
    dydt[15] = InR4-RR4;
    dydt[16] = RR4;
    dydt[17] = RI1;
    dydt[18] = RI2;
    dydt[19] = RI3;
    dydt[20] = RI4;
    return dydt;
  }
}
data {
  int<lower = 1> n_obs; // Number of days sampled
  int<lower = 1> n_params; // Number of model parameters
  int<lower = 1> n_difeq; // Number of differential equations in the system
  int y1[n_obs];
  int y2[n_obs];
  int y3[n_obs];
  int y4[n_obs];
  real t0; // Initial time point (zero)
  real ts[n_obs]; // Time points that were sampled
  vector[n_difeq] y0;
}
parameters {
  real<lower = 0> params[n_params]; // Model parameters
  real<lower = 0, upper = 1> rho;
}
transformed parameters{
  vector[n_difeq] y_hat[n_obs]; // Output from the ODE solver
  real incidence1[n_obs];
  real incidence2[n_obs];
  real incidence3[n_obs];
  real incidence4[n_obs];
  y_hat = ode_rk45(SEIR_matrix_sym, y0, t0, ts, params);
  incidence1[1] =  rho * (y_hat[1, 17]  - y0[17]);
  incidence2[1] =  rho * (y_hat[1, 18]  - y0[18]);
  incidence3[1] =  rho * (y_hat[1, 19]  - y0[19]);
  incidence4[1] =  rho * (y_hat[1, 20]  - y0[20]);
  for (i in 1:n_obs-1) {
    incidence1[i + 1] = rho * (y_hat[i + 1, 17] - y_hat[i, 17] + 1e-5);
    incidence2[i + 1] = rho * (y_hat[i + 1, 18] - y_hat[i, 18] + 1e-5);
    incidence3[i + 1] = rho * (y_hat[i + 1, 19] - y_hat[i, 19] + 1e-5);
    incidence4[i + 1] = rho * (y_hat[i + 1, 20] - y_hat[i, 20] + 1e-5);
   }
}
model {
    params ~ normal(0, 10);
    rho    ~ normal(0.5, 0.5);
    y1     ~ poisson(incidence1);
    y2     ~ poisson(incidence2);
    y3     ~ poisson(incidence3);
    y4     ~ poisson(incidence4);
}
generated quantities {
  real log_lik;
  log_lik = poisson_lpmf(y1 | incidence1) + poisson_lpmf(y2 | incidence2) + poisson_lpmf(y3 | incidence3) + poisson_lpmf(y4 | incidence4);
}
