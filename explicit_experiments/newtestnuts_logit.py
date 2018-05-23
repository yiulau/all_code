import pickle
import time
import os
import numpy
import pandas as pd
import pystan
import torch
from explicit.general_util import logsumexp_torch
from explicit.leapfrog_ult_util import leapfrog_ult as leapfrog
from torch.autograd import Variable

from explicit.nuts_util import NUTS
seedid = 30
numpy.random.seed(seedid)
torch.manual_seed(seedid)
dim = 5
num_ob = 100
chain_l = 1000
burn_in = 100
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

def T(p):
    return(torch.dot(p,p)*0.5)

def H(q,p,return_float):
    if return_float:
        return((V(q)+T(p)).data[0])
    else:
        return((V(q)+T(p)))




#q = Variable(torch.randn(dim),requires_grad=True)

v = -1

epsilon = 0.11

store = torch.zeros((chain_l,dim))
begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    out = NUTS(q,0.12,H,leapfrog,max_tdepth)
    store[i,] = out[0].data # turn this on when using Nuts
    q.data = out[0].data # turn this on when using nuts

total = time.time() - begin
print("total time is {}".format(total))
print("length of chain is {}".format(chain_l))
print("length of burn in is {}".format(burn_in))
print("Use logit")
store = store[burn_in:,]
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
#print(empCov)
print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
print("mean is {}".format(emmean))
print(fit)