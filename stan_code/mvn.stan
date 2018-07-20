data {

  int K ;
  matrix[K,K] precision; //the model matrix
}
parameters {
  vector[K] beta; //the regression parameters
}

model {
  target += -b;
  //y ~ normal(linpred,1);
}
