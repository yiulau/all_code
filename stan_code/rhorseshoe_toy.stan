data {
  int N; //the number of observations
  real y[N]; //the response

}
parameters {
   //the regression parameters
  vector<lower=0>[N] local_r1;
  vector<lower=0>[N] local_r2;
  real<lower=0> global_r1;
  real<lower=0> global_r2;
  vector[N] z;
  real<lower=0> caux;
}

transformed parameters{
  vector<lower=0>[N] lamb = local_r1 .* sqrt(local_r2);
  real<lower=0> tau = global_r1 * sqrt(global_r2);
  real<lower=0> c = sqrt(caux) ;
  vector<lower=0>[N] lamb_tilde = sqrt( c^2 * square(lamb) ./ (c^2 + tau^2*square(lamb)) );
  vector[N] beta = z .* lamb_tilde * tau;
}

model {
  z ~ normal(0,1);
  local_r1 ~ normal(0,1);
  global_r1 ~ normal(0,1);
  local_r2 ~ inv_gamma(0.5,0.5);
  global_r2 ~ inv_gamma(0.5,0.5);
  caux ~ inv_gamma(2,8);
  y ~ normal(0,beta);
}