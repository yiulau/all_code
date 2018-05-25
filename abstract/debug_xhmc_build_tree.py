import pickle
import time
import os
import numpy
import pandas as pd
import pystan
import torch
from explicit.general_util import logsumexp_torch
from explicit.leapfrog_ult_util import leapfrog_ult
from torch.autograd import Variable
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_class_point import point
from abstract.abstract_nuts_util import abstract_NUTS_xhmc
from explicit.nuts_util import NUTS_xhmc
seedid = 30
numpy.random.seed(seedid)
torch.manual_seed(seedid)
dim = 5
num_ob = 100
chain_l = 1000
burn_in = 100
max_tdepth = 10

stan_sampling = False
y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
X_np = numpy.random.randn(num_ob,dim)
address = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data/pima_india.csv"
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



y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)
inputq = torch.randn(dim)

inputp = torch.randn(dim)
q = Variable(inputq.clone(),requires_grad=True)
p = Variable(inputp.clone())

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

def dG_dt(q,p):
    # q_grad is dU/dq
    # exact form depends on the momentum distribution ika the metric
    H_fun = H(q, p,False)
    H_fun.backward()
    q_grad = q.grad.data.clone()
    q.grad.data.zero_()
    return(torch.dot(p.data,p.data) - torch.dot(q.data,q_grad))

print("exact H {}".format(H(q,p,True)))
print("exact V {}".format(V(q).data[0]))
print("exact T {}".format(T(p).data[0]))
v_obj = V_pima_inidan_logit()
metric_obj = metric("unit_e",v_obj)
Ham = Hamiltonian(v_obj,metric_obj)
#q_point = Ham.V.q_point.point_clone()
#p_point = Ham.T.p_point.point_clone()
q_point = point(V=Ham.V)
p_point = point(T=Ham.T)
#print(hex(id(q_point.flattened_tensor)))
#print(hex(id(q_point.list_tensor[0])))

q_point.flattened_tensor.copy_(inputq)
p_point.flattened_tensor.copy_(inputp)
#print(q_point.flattened_tensor)
#print(q_point.list_tensor[0])
#exit()
q_point.load_flatten()
p_point.load_flatten()
print("abstract H {}".format(Ham.evaluate(q_point,p_point)))
print("abstract T {}".format(Ham.T.evaluate_scalar(p_point=p_point)))
print("abstract V {}".format(Ham.V.evaluate_scalar()))
print("input q diff{}".format((q.data-q_point.flattened_tensor).sum()))
print("input p diff {}".format((p.data-p_point.flattened_tensor).sum()))

debug_dict = {"explicit":None,"abstract":None}