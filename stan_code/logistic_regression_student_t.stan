data {
  int N; //the number of observations
  int<lower=0,upper=1> y[N]; //the response
  int K;
  matrix[N,K] X;
}
parameters {
  real<lower=0> local_r2;
  vector[K] z;
}
transformed parameters{
  vector[K] beta = z * sqrt(local_r2);
}
model {
  z ~ normal(0,1);
  local_r2 ~ inv_gamma(0.5,0.5);
  y ~ bernoulli_logit(X*beta);
}