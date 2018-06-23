data {
  int N; //the number of observations
  real y[N]; //the response
  int K;
  matrix[N,K] X;
}
parameters {
  real<lower=0> local_r2;
  vector[K] z;
}

transformed parameters{
  vector[K] beta = z *local_r2;
}

model {
  z ~ normal(0,1);
  local_r2 ~ inv_gamma(0.5,0.5);
  y ~ normal(X*beta,1);
}