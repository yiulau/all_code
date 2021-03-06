import pickle
import time
import os
import numpy
import pandas as pd
import pystan
import torch
from explicit.general_util import logsumexp_torch
from explicit.genleapfrog_ult_util import getH, eigen, softabs_map, dtaudp, generalized_leapfrog
from torch.autograd import Variable
from experiments.correctdist_experiments.prototype import check_mean_var

from explicit.nuts_util import GNUTS
seedid = 30
numpy.random.seed(seedid)
torch.manual_seed(seedid)
dim = 5
num_ob = 100
chain_l = 10
burn_in = 0
max_tdepth = 10

stan_sampling = True

y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
X_np = numpy.random.randn(num_ob,dim)
#address = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data/pima_india.csv"
address = os.environ["PYTHONPATH"] + "/input_data/pima_india.csv"
df = pd.read_csv(address,header=0,sep=" ")
#print(df)
dfm = df.as_matrix()
#print(dfm)
#print(dfm.shape)
y_np = dfm[:,8]
y_np = y_np.astype(numpy.int64)
X_np = dfm[:,1:8]
dim = X_np.shape[1]
num_ob = X_np.shape[0]
data = dict(y=y_np,X=X_np,N=num_ob,p=dim)
if stan_sampling:
    recompile = False
    if recompile:
        mod = pystan.StanModel(file="./alt_log_reg.stan")
        with open('model.pkl', 'wb') as f:
            pickle.dump(mod, f)
    else:
        mod = pickle.load(open('model.pkl', 'rb'))

    fit = mod.sampling(data=data, refresh=0)

#print(fit)

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim))



def V(beta):
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(logsumexp_torch(Variable(torch.zeros(num_ob)), torch.mv(X, beta)))
    prior = -torch.dot(beta, beta) * 0.5
    posterior = prior + likelihood
    return(-posterior)

def T(q,alpha):
    def T_givenq(p):
        _,H_ = getH(q,V)
        out = eigen(H_.data)
        lam = out[0]
        Q = out[1]
        temp = softabs_map(lam,alpha)
        inv_exp_H = torch.mm(torch.mm(Q,torch.diag(1/temp)),torch.t(Q))
        o = 0.5 * torch.dot(p.data,torch.mv(inv_exp_H,p.data))
        temp2 = 0.5 * torch.log((temp)).sum()
        return(o + temp2)
    return(T_givenq)

def pi_wrap(alpha,return_float):
    def inside(x,y,return_float):
        return(H(x,y,alpha,return_float))
    return inside
def H(q,p,alpha,return_float):
    if return_float:
        return(V(q).data[0] + T(q,alpha)(p))
    else:
        return(V(q) + Variable(T(q,alpha)(p)))

def genleapfrog_wrap(alpha,delta,V):
    def inside(q,p,ep,pi):
        return generalized_leapfrog(q,p,ep,alpha,delta,V)
    return(inside)

def p_sharp(q,p):
    _,H_ = getH(q, V)
    lam, Q = eigen(H_.data)
    p_s = dtaudp(p.data, alp, lam, Q)
    return(p_s)

alp = 1e6
fi_fake = pi_wrap(alp,True)
gleapfrog = genleapfrog_wrap(alp,0.1,V)

store = torch.zeros((chain_l,dim))
begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    out = GNUTS(q,0.1,fi_fake,gleapfrog,10,p_sharp)
    store[i,] = out[0].data # turn this on when using Nuts
    q.data = out[0].data # turn this on when using nuts
    print("q {}".format(q.data))

total = time.time() - begin
print("total time is {}".format(total))
print("length of chain is {}".format(chain_l))
print("length of burn in is {}".format(burn_in))
print("Use logit")
store = store[burn_in:,]
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
#print("store is {}".format(store))
#print(empCov)
print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
print("mean is {}".format(emmean))
if stan_sampling:
    print(fit)

mcmc_samples = store

address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain.pkl"
correct = pickle.load(open(address, 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()

output = check_mean_var(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = output["mcmc_mean"],output["mcmc_Cov"]
pc_mean,pc_cov = output["pc_of_mean"],output["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)