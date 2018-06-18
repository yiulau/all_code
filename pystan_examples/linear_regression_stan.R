library(rstan)
library(reticulate)

datasets = import("sklearn.datasets")
out = datasets$load_boston()
X = out$data
y = out$target

X = scale(X)
#print(mean(X[,1]))


N = dim(X)[1]
K = dim(X)[2]
data = list(X=X,y=y,N=N,K=K)

address = "/home/yiulau/PycharmProjects/all_code/stan_code/linear_regression.stan"
model = stan_model(file=address)
options(mc.cores = parallel::detectCores())

o1 = sampling(model,data=list(y=y,X=X,N=N,K=K),algorithm="HMC",control=list(metric="unit_e",adapt_engaged=TRUE))

o2 = sampling(model,data=list(y=y,X=X,N=N,K=K),algorithm="HMC",control=list(metric="unit_e",adapt_engaged=TRUE,int_time=1.))

o3 = sampling(model,data=list(y=y,X=X,N=N,K=K),algorithm="HMC",control=list(metric="diag_e",adapt_engaged=TRUE,int_time=1.,stepsize=0.1))
o4 = sampling(model,data=list(y=y,X=X,N=N,K=K),algorithm="HMC",control=list(metric="unit_e",int_time=5))

o5 = sampling(model,data=list(y=y,X=X,N=N,K=K),algorithm="HMC",control=list(int_time=1))

o6 = sampling(model,data=list(y=y,X=X,N=N,K=K),control=list(metric="unit_e"))

o7 = sampling(model,data=list(y=y,X=X,N=N,K=K),control=list(metric="diag_e"))

o8 = sampling(model,data=list(y=y,X=X,N=N,K=K))

print(o1)
print(o2)
print(o3)
print(o4)
print(o5)
print(o6)
print(o7)
print(o8)
sampler_params <- get_sampler_params(o8, inc_warmup = TRUE)
sampler_params_chain1 <- sampler_params[[4]]
colnames(sampler_params_chain1)

print(sampler_params_chain1[1:900,"stepsize__"])

print(sampler_params_chain1[1:900,"n_leapfrog__"])
print(sampler_params_chain1[1000:1900,"n_leapfrog__"])

print(sum(sampler_params_chain1[1000:1900,"n_leapfrog__"]))




