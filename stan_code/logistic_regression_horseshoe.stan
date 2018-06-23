data {
  int N; //the number of observations
  int<lower=0,upper=1> y[N]; //the response
  int K;
  matrix[N,K] X;
}
parameters {
  vector<lower=0>[K] local_r1;
  vector<lower=0>[K] local_r2;
  real<lower=0> global_r1;
  real<lower=0> global_r2;
  vector[K] z;
}
transformed parameters {
  vector[K] beta;
  vector<lower=0>[K] lamb;
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
  y ~ bernoulli_logit(X*beta);
}