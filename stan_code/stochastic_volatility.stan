data {
    int N; # number of obs

    real y[N]; # 0-1 response

}
parameters {
    vector[N] x;
    real beta;
    real unconstrained_phi;
    real unconstrained_sigma;

}


model{
    phi = inv_logit(unconstrained_phi)
    sigma = exp(unconstrained_sigma)
    y_var = exp(x*0.5)*beta * exp(x*0.5)*beta
    target+=
}
