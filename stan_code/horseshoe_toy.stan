data {
  int N; //the number of observations
  real y[N]; //the response

}
parameters {
  vector<lower=0>[N] local_r1;
  vector<lower=0>[N] local_r2;
  real<lower=0> global_r1;
  real<lower=0> global_r2;
  vector[N] z;
}

transformed parameters{
  vector<lower=0> [N] lamb;
  vector [N] beta;
  real<lower=0> tau;
  lamb = local_r1 .* sqrt(local_r2);
  tau = global_r1 * sqrt(global_r2);
  beta = z .* lamb * tau;
}

model {
  z ~ normal(0,1);
  local_r1 ~ normal(0,1);
  global_r1 ~ normal(0,1);
  local_r2 ~ inv_gamma(0.5,0.5);
  global_r2 ~ inv_gamma(0.5,0.5);
  y ~ normal(beta,1);

}
