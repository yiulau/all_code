# check that abstract leapfrog and explicit leapfrog gives the same answer
import numpy,math
import pandas as pd
import torch
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_genleapfrog_ult_util import generalized_leapfrog as abstract_generalized_leapfrog
from abstract.metric import metric
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from torch.autograd import Variable
from abstract.abstract_class_point import point
from explicit.genleapfrog_ult_util import generalized_leapfrog as explicit_generalized_leapfrog
from explicit.genleapfrog_ult_util import getH, eigen, softabs_map,generate_momentum,rmhmc_step
import os

seedid = 33
numpy.random.seed(seedid)
torch.manual_seed(seedid)
#y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
#X_np = numpy.random.randn(num_ob,dim)
#source_root = os.environ["PYTHONPATH"]
#print(source_root)
#address = source_root+"/input_data/pima_india.csv"
address = os.environ["PYTHONPATH"] + "/input_data/pima_india.csv"
#address = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data/pima_india.csv"
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
p = Variable(inputp.clone(),requires_grad=False)

def V(q):
    beta = q
    likelihood = torch.dot(beta,torch.mv(torch.t(X),y)) - \
    torch.sum(torch.log(1+torch.exp(torch.mv(X,beta))))
    prior = -torch.dot(beta,beta) * 0.5
    posterior = prior + likelihood
    return(-posterior)


def T(q,alpha):
    def T_givenq(p):
        _,H_ = getH(q,V)
        #debug_dict.update({"explicit":_.data.clone()})
        out = eigen(H_.data)
        lam = out[0]
        Q = out[1]
        temp = softabs_map(lam,alpha)
        #print("explicit p {}".format(q.data))
        inv_exp_H = torch.mm(torch.mm(Q,torch.diag(1/temp)),torch.t(Q))
        o = 0.5 * torch.dot(p.data,torch.mv(inv_exp_H,p.data))
        temp2 = 0.5 * torch.log((temp)).sum()
        print("explicit tau {}".format(o))
        print("explicit logdetmetric {}".format(temp2))
        return(o + temp2)
    return(T_givenq)

def H(q,p,alpha):
    # returns float
    return(V(q).data[0] + T(q,alpha)(p))

alpha = 1e6
# for i in range(10):
#     q = Variable(q.data.clone(), requires_grad=True)
#     _, H_ = getH(q, V)
#     lam, Q = eigen(H_.data)
#     p = Variable(generate_momentum(alpha,lam,Q))
#     current_H = H(q,p,alpha)
#     print("start H {}".format(current_H))
#     for _ in range(10):
#         out = explicit_generalized_leapfrog(q,p,0.1,alpha,0.1,V)
#         q.data = out[0].data
#         p.data = out[1].data
#
#     proposed_H = H(q,p,alpha)
#     print("end H {}".format(proposed_H))
#
#     accept_rate = math.exp(min(0,current_H - proposed_H))
#     u = numpy.random.rand(1)
#     if u < accept_rate:
#         next_q = q
#         accepted = True
#     else:
#         next_q =
#         accepted = False
#     print("accept_rate {}".format(accept_rate))
#
# exit()
# chain_l=100
# burn_in = 10
# store = torch.zeros((chain_l,dim))
#
# #g,H_ = getH(q,V)
# #lam,Q = eigen(H_.data)
# q = Variable(inputq.clone(),requires_grad=True)
# for i in range(chain_l):
#     print("round {}".format(i))
#     out = rmhmc_step(q,H,0.1,10,alpha,0.1,V)
#     store[i,]=out[0].data
#     print("accepted_rate {}".format(out[2]))
#     print("accepted {}".format(out[3]))
#     q.data = out[0].data
#     print("q {}".format(q.data))
#
# store = store[burn_in:, ]
# store = store.numpy()
# empCov = numpy.cov(store, rowvar=False)
# emmean = numpy.mean(store, axis=0)
# # print(empCov)
# # print("store is {}".format(store))
# print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
# print("mean is {}".format(emmean))
# exit()
for i in range(10):
    print("round {}".format(i))
    out = rmhmc_step(q,H,0.1,10,alpha,0.1,V)
    #store[i,]=out[0].data
    print("accepted_rate {}".format(out[2]))
    print("accepted {}".format(out[3]))
    q.data = out[0].data
    print("q {}".format(q.data))

#out = rmhmc_step(q,H,0.1,10,alpha,0.1,V)