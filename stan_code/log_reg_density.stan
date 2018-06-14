data {
    int N; # number of obs
    int p; # dimension of model. num of covariates
    real y[N]; # 0-1 response
    matrix[N,p] X; #design matrix
}
parameters {
    vector[p] beta;
}

model{
    target+= dot_product(to_vector(y),log(inv_logit(X*beta)))+dot_product(1-to_vector(y),log(1-inv_logit(X*beta)));
}

