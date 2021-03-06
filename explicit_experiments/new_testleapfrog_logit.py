import pickle
import os,time
import numpy
import pandas as pd
import pystan
import torch
from explicit.general_util import logsumexp_torch
from torch.autograd import Variable
from explicit.leapfrog_ult_util import HMC_alt_ult, leapfrog_ult
from experiments.correctdist_experiments.prototype import check_mean_var

seedid = 30
numpy.random.seed(seedid)
torch.manual_seed(seedid)

dim = 4
num_ob = 25
chain_l = 500
burn_in = 100
stan_sampling = False



#y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
#X_np = numpy.random.randn(num_ob,dim)
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

#correct_samples = fit.extract(permuted=True)["beta"]

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim),requires_grad=False)

def V(beta):
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(logsumexp_torch(Variable(torch.zeros(num_ob)), torch.mv(X, beta)))
    prior = -torch.dot(beta, beta) * 0.5
    posterior = prior + likelihood
    return(-posterior)

def T(p):
    return(torch.dot(p,p)*0.5)


def H(q,p,return_float):
    if return_float:
        return((V(q)+T(p)).data[0])
    else:
        return((V(q)+T(p)))

def generate_momentum(q):
    return(torch.randn(len(q)))
store = torch.zeros((chain_l,dim))
start_time = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    #out = HMC_alt_ult(0.1,10,q,leapfrog_ult,H,False)
    out = HMC_alt_ult(epsilon=0.1,L=10,current_q=q,leapfrog=leapfrog_ult,H_fun=H,generate_momentum=generate_momentum)
    store[i,]=out[0].data
    q.data = out[0].data

end_time = time.time()

store = store[burn_in:,]
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
print("length of chain is {}".format(chain_l))
print("burn in is {}".format(burn_in))
#print(empCov)
print("total time is {}".format(end_time))
print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
print("mean is {}".format(emmean))
mcmc_samples=store

if stan_sampling:
    print(fit)

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