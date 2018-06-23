data {
  int N; //the number of observations
  real y[N]; //the response

}
parameters {
  real<lower=0> sigma2;
  vector[N] z;
}

transformed parameters{
  vector[N] beta = z * sqrt(sigma2);
}

model {
  z ~ normal(0,1);
  sigma2 ~ inv_gamma(0.5,0.5);
  y ~ normal(beta,1);
}