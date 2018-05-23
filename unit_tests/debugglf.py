# check that abstract leapfrog and explicit leapfrog gives the same answer
import numpy
import pandas as pd
import torch
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_genleapfrog_ult_util import generalized_leapfrog as abstract_generalized_leapfrog
from abstract.metric import metric
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from torch.autograd import Variable
from abstract.abstract_class_point import point
from explicit.genleapfrog_ult_util import generalized_leapfrog as explicit_generalized_leapfrog
from explicit.genleapfrog_ult_util import getH, eigen, softabs_map
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
q = Variable(inputq,requires_grad=True)
p = Variable(inputp,requires_grad=False)

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
#debug_dict = {"abstract":None,"explicit":None}

#debug_dict.update({"explicit":y.data.clone()})
# first verify they have the same Hamiltonian function
print("exact H {}".format(H(q,p,alpha)))
print("exact V {}".format(V(q).data[0]))
print("exact T {}".format((T(q,alpha)(p))))
v_obj = V_pima_inidan_logit()
metric_obj = metric("softabs",v_obj,alpha)
Ham = Hamiltonian(v_obj,metric_obj)
q_point = point(V=Ham.V)
p_point = point(T=Ham.T)

q_point.flattened_tensor.copy_(inputq)
p_point.flattened_tensor.copy_(inputp)
q_point.load_flatten()
p_point.load_flatten()
#print("q point need_flatten {}".format(q_point.need_flatten))
#print("q_point syncro {}".format(q_point.assert_syncro()))
#print("q_point list_tensor {}".format(q_point.list_tensor))
#print("q_point flattened_tensor {}".format(q_point.flattened_tensor))
print("abstract H {}".format(Ham.evaluate(q_point,p_point)))
print("abstract V {}".format(Ham.V.evaluate_scalar(q_point)))
print("abstract T {}".format(Ham.T.evaluate_scalar(q_point,p_point)))
print("input q diff{}".format((q.data-q_point.flattened_tensor).sum()))
print("input p diff {}".format((p.data-p_point.flattened_tensor).sum()))

#debug_dict.update({"abstract":Ham.V.y.data.clone()})
#diff_term = ((debug_dict["abstract"]-debug_dict["explicit"])*(debug_dict["abstract"]-debug_dict["explicit"])).sum()
#print("H_diff {}".format(diff_term))



for i in range(10):
    outq, outp = explicit_generalized_leapfrog(q, p, 0.1, alpha, 0.1, V)
    outq_a, outp_a, stat = abstract_generalized_leapfrog(q_point, p_point, 0.1, Ham)
    q,p = outq,outp
    q_point,p_point = outq_a,outp_a
#outq,outp = explicit_generalized_leapfrog(q,p,0.1,alpha,0.1,V)
#outq_a,outp_a,stat = abstract_generalized_leapfrog(q_point,p_point,0.1,Ham)
# compare dV
#print(debug_dict)

#diff_dV = ((debug_dict["abstract"]-debug_dict["explicit"])*(debug_dict["abstract"]-debug_dict["explicit"])).sum()
#print(diff_dV)
#exit()


diffq = ((outq.data - outq_a.flattened_tensor)*(outq.data - outq_a.flattened_tensor)).sum()
diffp = ((outp.data - outp_a.flattened_tensor)*(outp.data - outp_a.flattened_tensor)).sum()
print("diff outq {}".format(diffq))
print("diff outp {}".format(diffp))
print("end H explicit{}".format(H(q,p,alpha)))
print("end H abstract {}".format(Ham.evaluate(q_point,p_point)))

exit()
#print(outp.data)
#print(outp_a.flattened_tensor)
print("exact")
print("q {}".format(outq))
print("p {}".format(outp))

print("abstract")


print("q {}".format(q_point.flattened_tensor))
print("p {}".format(p_point.flattened_tensor))
print(stat.divergent)