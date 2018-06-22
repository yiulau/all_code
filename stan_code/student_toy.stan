data {
  int N; //the number of observations
  real y[N]; //the response

}
parameters {
  vector[N] beta; //the regression parameters
  vector<lower=0>[N] local_r1
  vector<lower=0>[N] local_r2
  vector[N] = z
}

trasnformed parameters{
  lamb = local_r1 * sqrt(local_r2)
  beta = z * lamb
}

model {
  z ~ N(0,1)
  local_r1 ~ N(0,1)
  local_r2 ~ inv_gamma(0.5,0.5)
  y ~ normal(0,beta)
}