data {
  int N; //the number of observations
  int K; //the number of columns in the model matrix
  real y[N]; //the response
  matrix[N,K] X; //the model matrix
}
parameters {
  vector[K] beta; //the regression parameters
}
//transformed parameters {
//  vector[N] linpred;
//  linpred <- X*beta;
//}
model {
  target += -dot_product(to_vector(y) - X*beta,to_vector(y)-X*beta)*0.5 -dot_product(beta,beta)*0.5;
  //y ~ normal(linpred,1);
}
