data {
  int N; //the number of observations
  int y[N]; //the response
  int K;
  matrix[N,K] X;
}
parameters {
  vector<lower=0>[K] local_r1;
  vector<lower=0>[K] local_r2;
  real<lower=0> global_r1;
  real<lower=0> global_r2;
  vector[K] z;
  real<lower=0> caux;
}
transformed parameters{
  vector<lower=0>[K] lamb = local_r1 .* sqrt(local_r2);
  real<lower=0> tau = global_r1 * sqrt(global_r2);
  real<lower=0> c = sqrt(caux) * 2;
  vector<lower=0>[K] lamb_tilde = sqrt( c^2 * square(lamb)./(c^2 + tau^2*square(lamb)));
  vector[K] beta = z .* lamb_tilde * tau;
}
model {
  z ~ normal(0,1);
  local_r1 ~ normal(0,1);
  global_r1 ~ normal(0,1);
  local_r2 ~ inv_gamma(0.5,0.5);
  global_r2 ~ inv_gamma(0.5,0.5);
  caux ~ inv_gamma(0.5*4,0.5*4);
  y ~ bernoulli_logit(X*beta);
}