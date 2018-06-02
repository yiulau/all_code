# check that abstract leapfrog and explicit leapfrog gives the same answer
import numpy
import pandas as pd
import torch
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_leapfrog_util import abstract_leapfrog_ult
from abstract.metric import metric
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from explicit.general_util import logsumexp_torch
from torch.autograd import Variable
from abstract.abstract_class_point import point
from explicit.leapfrog_ult_util import leapfrog_ult
#y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
#X_np = numpy.random.randn(num_ob,dim)
seedid = 301321
numpy.random.seed(seedid)
torch.manual_seed(seedid)
import os
#print(os.getcwd())
import sys
#print(sys.path)
print(os.environ["PYTHONPATH"])
#address = "/home/yiulau/work/thesis_code/explain_hmc/input_data/pima_india.csv"
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


input_data = {"X_np":X_np,"y_np":y_np}

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)


inputq = torch.randn(dim)

inputp = torch.randn(dim)
q = Variable(inputq.clone(),requires_grad=True)
p = Variable(inputp.clone(),requires_grad=False)

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

# first verify they have the same Hamiltonian function
# print("exact H {}".format(H(q,p,True)))
# print("exact V {}".format(V(q).data[0]))
# print("exact T {}".format(T(p).data[0]))
#v_obj = V_pima_inidan_logit()
from distributions.neural_nets.fc_V_hierarchical import V_fc_test_hyper
v_obj = V_fc_test_hyper(input_data)
metric_obj = metric("unit_e",v_obj)
Ham = Hamiltonian(v_obj,metric_obj)
#q_point = Ham.V.q_point.point_clone()
#p_point = Ham.T.p_point.point_clone()
q_point = point(V=Ham.V)
p_point = point(T=Ham.T)
#print(hex(id(q_point.flattened_tensor)))
#print(hex(id(q_point.list_tensor[0])))

#q_point.flattened_tensor.copy_(inputq)
#p_point.flattened_tensor.copy_(inputp)
q_point.flattened_tensor.normal_()
p_point.flattened_tensor.normal_()
#print(q_point.flattened_tensor)
#print(q_point.list_tensor[0])
#exit()
q_point.load_flatten()
p_point.load_flatten()


print("abstract H {}".format(Ham.evaluate(q_point,p_point)))
#print("abstract T {}".format(Ham.T.evaluate_scalar(p_point)))
#print("abstract V {}".format(Ham.V.evaluate_scalar(q_point)))
# print("input q diff{}".format((q.data-q_point.flattened_tensor).sum()))
# print("input p diff {}".format((p.data-p_point.flattened_tensor).sum()))

start_q = q_point.point_clone()

L = 100
for i in range(L):
    #print(i)
    #outq,outp = leapfrog_ult(q,p,0.1,H)
    outq_a,outp_a,stat = abstract_leapfrog_ult(q_point,p_point,0.1,Ham)
    #q, p = outq, outp
    q_point, p_point = outq_a, outp_a
    print("H {}".format(Ham.evaluate(q_point, p_point)))

#diffq = ((outq.data - outq_a.flattened_tensor)*(outq.data - outq_a.flattened_tensor)).sum()
#diffp = ((outp.data - outp_a.flattened_tensor)*(outp.data - outp_a.flattened_tensor)).sum()
print("abstract end H {}".format(Ham.evaluate(q_point,p_point)))

prob = v_obj.predict()
prediction = (prob>0.5).type("torch.FloatTensor")
#print(prediction)
target = v_obj.y.data.type("torch.FloatTensor")

print(sum(prediction.numpy()==target.numpy()))
#print(v_obj.y.type("torch.FloatTensor"))
#print("diff outq {}".format(diffq))
#print("diff outp {}".format(diffp))


#print("exact")
#print("q {}".format(outq))
#print("p {}".format(outp))

#print("abstract")


#print("q {}".format(q_point.flattened_tensor))
#print("p {}".format(p_point.flattened_tensor))
