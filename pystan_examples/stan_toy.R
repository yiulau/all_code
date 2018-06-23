library(rstan)
num_obs = 100
num_non_zeros = 20 

y = rep(0,num_obs)
true_beta = rnorm(num_non_zeros) * 5
y[1:num_non_zeros] = true_beta
y = y + rnorm(num_obs)
print(true_beta)

address = "/home/yiulau/PycharmProjects/all_code/stan_code/horseshoe_toy.stan"
model_hs = stan_model(file=address)
address = "/home/yiulau/PycharmProjects/all_code/stan_code/rhorseshoe_toy.stan"
model_rhs = stan_model(file=address)
address = "/home/yiulau/PycharmProjects/all_code/stan_code/linear_regression_horseshoe.stan"
model_lr_hs = stan_model(file=address)
address = "/home/yiulau/PycharmProjects/all_code/stan_code/linear_regression_rhorseshoe.stan"
model_lr_rhs = stan_model(file=address)
address = "/home/yiulau/PycharmProjects/all_code/stan_code/linear_regression_student_t.stan"
model_lr_student = stan_model(file=address)
address = "/home/yiulau/PycharmProjects/all_code/stan_code/logistic_regression_horseshoe.stan"
model_logit_horseshoe = stan_model(file=address)
address = "/home/yiulau/PycharmProjects/all_code/stan_code/logistic_regression_rhorseshoe.stan"
model_logit_rhorseshoe = stan_model(file=address)
address = "/home/yiulau/PycharmProjects/all_code/stan_code/logistic_regression_student_t.stan"
model_logit_student = stan_model(file=address)
options(mc.cores = parallel::detectCores())

data = list(y=y,N=num_obs)
o1 = sampling(model_hs,data=data,control=list("metric"="diag_e",adapt_delta = 0.9))
print(o1)
beta_summary <- summary(o1, pars = c("beta"))$summary
lp_summary = summary(o1,pars=c("lp__"))$summary
tau_summary = summary(o1,pars=c("tau"))$summary
print(beta_summary[1:non_zero_dim,])
print(true_beta[1:non_zero_dim])
print(tau_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o1, inc_warmup = TRUE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)
####################################################################################################################################
o2 = sampling(model_hs,data=data,control=list("metric"="dense_e",adapt_delta = 0.99,max_treedepth=15))

beta_summary <- summary(o2, pars = c("beta"))$summary
lp_summary = summary(o2,pars=c("lp__"))$summary
tau_summary = summary(o2,pars=c("tau"))$summary
print(beta_summary)
print(tau_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o2, inc_warmup = TRUE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)
#######################################################################################################################################

o3 = sampling(model_rhs,data=data,control=list("metric"="diag_e",adapt_delta = 0.9,max_treedepth=10))

beta_summary <- summary(o3, pars = c("beta"))$summary
lp_summary = summary(o3,pars=c("lp__"))$summary
tau_summary = summary(o3,pars=c("tau"))$summary
print(beta_summary)
print(tau_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o3, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)

###################################################################################################################################
o4 = sampling(model_rhs,data=data,control=list("metric"="dense_e",adapt_delta = 0.9,max_treedepth=10))

beta_summary <- summary(o4, pars = c("beta"))$summary
lp_summary = summary(o4,pars=c("lp__"))$summary
tau_summary = summary(o4,pars=c("tau"))$summary
print(beta_summary)
print(tau_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o4, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)
################################################################################################################################
num_obs = 400
non_zero_dim = 20
full_p = 100
X = matrix(rnorm(num_obs*full_p),nrow=num_obs,ncol=full_p)
library(Matrix)
rankMatrix(X)
true_beta = rnorm(full_p)
true_beta[1:non_zero_dim] = rnorm(non_zero_dim)*5
print(true_beta[1:non_zero_dim])
y = X%*%true_beta + rnorm(num_obs)
y = drop(y)
data = list(y=y,X=X,N=num_obs,K=full_p)

o5 = sampling(model_lr_hs,data=data,control=list("metric"="diag_e",adapt_delta = 0.9,max_treedepth=10))

beta_summary <- summary(o5, pars = c("beta"))$summary

lp_summary = summary(o5,pars=c("lp__"))$summary
tau_summary = summary(o5,pars=c("tau"))$summary
print(beta_summary)
print(tau_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o5, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)

##################################################################################################################################
o6 = sampling(model_lr_rhs,data=data,control=list("metric"="diag_e",adapt_delta = 0.9,max_treedepth=15))

beta_summary <- summary(o6, pars = c("beta"))$summary
lp_summary = summary(o6,pars=c("lp__"))$summary
tau_summary = summary(o6,pars=c("tau"))$summary
c_summary <- summary(o6, pars = c("c"))$summary

print(beta_summary[1:non_zero_dim,])
print(true_beta[1:non_zero_dim])
print(tau_summary)
print(c_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o6, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)
#########################################################################################################################################
library(glmnet)
lambda <- 10^seq(10, -2, length = 100)
lasso.mod = glmnet(X, y, alpha = 1, lambda = lambda)
bestlam <- lasso.mod$lambda.min
lasso.coef  <- predict(lasso.mod, type = 'coefficients', s = bestlam)
out = glmnet(X, y, alpha = 1, lambda = 1)
print(coef(out))
##########################################################################################################################################
o7 = sampling(model_lr_student,data=data,control=list("metric"="diag_e",adapt_delta = 0.9,max_treedepth=15))

beta_summary <- summary(o7, pars = c("beta"))$summary
lp_summary = summary(o7,pars=c("lp__"))$summary
tau_summary = summary(o7,pars=c("tau"))$summary
c_summary <- summary(o7, pars = c("c"))$summary

print(beta_summary[1:non_zero_dim,])
print(true_beta[1:non_zero_dim])
print(tau_summary)
print(c_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o7, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)
#############################################################################################################################
# logistic 
n = 30 
dim = 100 
X = matrix(rnorm(n*dim),nrow=n,ncol=dim)
y = rep(0,n)
for(i in 1:n){
  y[i] = rbinom(n=1,size=1,prob=0.5)
  if(y[i]>0){
    X[i,1:2] = rnorm(2)*0.5 + 1
  }
  else{
    X[i,1:2] = rnorm(2)*0.5 -1
  }
}

data = list(X=X,y=y,N=n,K=dim)
o8 = sampling(model_logit_horseshoe,data=data,control=list("metric"="diag_e",adapt_delta = 0.9,max_treedepth=15))

beta_summary <- summary(o8, pars = c("beta"))$summary
lp_summary = summary(o8,pars=c("lp__"))$summary
tau_summary = summary(o8,pars=c("tau"))$summary
c_summary <- summary(o8, pars = c("c"))$summary

print(beta_summary[1:non_zero_dim,])
print(true_beta[1:non_zero_dim])
print(tau_summary)
print(c_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o8, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)

#######################################################################################################
o9 = sampling(model_logit_rhorseshoe,data=data,control=list("metric"="diag_e",adapt_delta = 0.99,max_treedepth=15))

beta_summary <- summary(o9, pars = c("beta"))$summary
lp_summary = summary(o9,pars=c("lp__"))$summary
tau_summary = summary(o9,pars=c("tau"))$summary
c_summary <- summary(o9, pars = c("c"))$summary

print(beta_summary[1:non_zero_dim,])
print(true_beta[1:non_zero_dim])
print(tau_summary)
print(c_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o9, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)
####################################################################################################

o10 = sampling(model_logit_student,data=data,control=list("metric"="diag_e",adapt_delta = 0.9,max_treedepth=15))

beta_summary <- summary(o10, pars = c("beta"))$summary
lp_summary = summary(o10,pars=c("lp__"))$summary
tau_summary = summary(o10,pars=c("tau"))$summary
c_summary <- summary(o10, pars = c("c"))$summary

print(beta_summary[1:non_zero_dim,])
print(true_beta[1:non_zero_dim])
print(tau_summary)
print(c_summary)
print(lp_summary)
sampler_params <- get_sampler_params(o10, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)
print(sampler_params_chain1[1:900,"stepsize__"])
print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])
num_divergent = sapply(sampler_params, function(x) sum(x[, "divergent__"]))
print(num_divergent)
############################################################################################################