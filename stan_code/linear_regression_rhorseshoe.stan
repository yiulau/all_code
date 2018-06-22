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
  real<lower=0> caux
}

trasnformed parameters{
  lamb = local_r1 * sqrt(local_r2)
  tau = global_r1 * sqrt(global_r2)
  c = sqrt(caux) * 2
  lamb_tilde = sqrt( c^2 * square(lamb) ./ (c^2 + tau^2*square(lamb)) )
  beta = lamb_tilde * tau

}

model {
  z ~ N(0,1)
  local_r1 ~ N(0,1)
  global_r1 ~ N(0,1)
  local_r2 ~ inv_gamma(0.5,0.5)
  global_r2 ~ inv_gamma(0.5,0.5)
  caux ~ inv_gamma(0.5*4,0.5*4)
  y ~ normal(X*beta,1)

}