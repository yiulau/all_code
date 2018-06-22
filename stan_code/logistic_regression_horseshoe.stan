data {
  int N; //the number of observations
  real y[N]; //the response
  int K;
  matrix[N,K] X;
}
parameters {
  vector[N] beta; //the regression parameters
  vector<lower=0>[N] local_r1
  vector<lower=0>[N] local_r2
  real<lower=0> global_r1
  real<lower=0> global_r2
  vector[N] = z
}

trasnformed parameters{
  lamb = local_r1 * sqrt(local_r2)
  tau = global_r1 * sqrt(global_r2)
  beta = lamb * tau
}

model {

  z ~ N(0,1)
  local_r1 ~ N(0,1)
  global_r1 ~ N(0,1)
  local_r2 ~ inv_gamma(0.5,0.5)
  global_r2 ~ inv_gamma(0.5,0.5)

  y ~ bernoulli_logit(X*beta)

}