# check that abstract leapfrog and explicit leapfrog gives the same answer
import numpy
import pandas as pd
import torch
from abstract.abstract_class_Ham import Hamiltonian
from abstract.deprecated_code.abstract_genleapfrog_ult_util import generalized_leapfrog as abstract_generalized_leapfrog
from abstract.metric import metric
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from torch.autograd import Variable
from abstract.abstract_class_point import point
from explicit.genleapfrog_ult_util import generalized_leapfrog as explicit_generalized_leapfrog
from explicit.genleapfrog_ult_util import getH, eigen, softabs_map
import os
torch.set_default_tensor_type("torch.DoubleTensor")
seedid = 333
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




y = Variable(torch.from_numpy(y_np),requires_grad=False).type("torch.DoubleTensor")

X = Variable(torch.from_numpy(X_np),requires_grad=False)



inputq = torch.randn(dim)

q = Variable(inputq,requires_grad=True)


# def V(q):
#     beta = q
#     likelihood = torch.dot(beta,torch.mv(torch.t(X),y)) - \
#     torch.sum(torch.log(1+torch.exp(torch.mv(X,beta))))
#     prior = -torch.dot(beta,beta) * 0.5
#     posterior = prior + likelihood
#     return(-posterior)

alpha = 1e6
def generate_momentum(q):
    # if lam == None or Q == None:
    #    H_ = self.linkedV.getH_tensor()
    #    lam, Q = eigen(H_)
    _, H_ = getH(q,V)
    lam, Q = eigen(H_.data)
    # print(lam)
    # print(Q)
    #exit()
    # print(lam.shape)
    # print(type(lam))
    # print(type(lam[0]))
    # exit()
    #print(lam)
    #exit()
    temp = torch.mm(Q, torch.diag(torch.sqrt(softabs_map(lam, alpha))))
    out = temp.mv(torch.randn(len(lam)))
    # print(temp)
    # exit()
    return (out)
n = 10
def V(q):
    # returns -log posterior
    x = q[:(n-1)]
    y = q[n-1]
    logp_y = -y*y * 1/9.* 0.5
    #print(self.beta.data)
    #print("bottom is {}".format(torch.exp(y*0.5)*torch.exp(y*0.5)))
    logp_x = -(torch.dot(x,x))/(torch.exp(y)) * 0.5 -0.5*(n-1)*y
    #print("x is {}".format(x))
    #print("p_x is {}".format(p_x))
    logprob = logp_y + logp_x
    out = -logprob

    #print("logpx {}".format(logp_x.data))
    #print("-logprob {}".format(out.data))
    return(out)

def T(q,alpha):
    def T_givenq(p):
        _,H_ = getH(q,V)
        #debug_dict.update({"explicit":_.data.clone()})
        out = eigen(H_.data)
        lam = out[0]
        Q = out[1]
        temp = softabs_map(lam,alpha)
        #print("softabs lam {}".format(temp))
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


dim = 10
inputq = torch.randn(dim)
inputq[dim-1]=3.1
q = Variable(inputq,requires_grad=True)
inputp = generate_momentum(q)

p = Variable(inputp,requires_grad=False)

#debug_dict = {"abstract":None,"explicit":None}

#debug_dict.update({"explicit":y.data.clone()})
# first verify they have the same Hamiltonian function
print("exact H {}".format(H(q,p,alpha)))
print("exact V {}".format(V(q).data[0]))
print("exact T {}".format((T(q,alpha)(p))))
v_obj = V_pima_inidan_logit()
#metric_obj = metric("softabs",v_obj,alpha)
#Ham = Hamiltonian(v_obj,metric_obj)
#q_point = point(V=Ham.V)
#p_point = point(T=Ham.T)

#q_point.flattened_tensor.copy_(inputq)
#p_point.flattened_tensor.copy_(inputp)
#q_point.load_flatten()
#p_point.load_flatten()
#print("q point need_flatten {}".format(q_point.need_flatten))
#print("q_point syncro {}".format(q_point.assert_syncro()))
#print("q_point list_tensor {}".format(q_point.list_tensor))
#print("q_point flattened_tensor {}".format(q_point.flattened_tensor))
# print("abstract H {}".format(Ham.evaluate(q_point,p_point)))
# print("abstract V {}".format(Ham.V.evaluate_scalar(q_point)))
# print("abstract T {}".format(Ham.T.evaluate_scalar(q_point,p_point)))
# print("input q diff{}".format((q.data-q_point.flattened_tensor).sum()))
# print("input p diff {}".format((p.data-p_point.flattened_tensor).sum()))

#debug_dict.update({"abstract":Ham.V.y.data.clone()})
#diff_term = ((debug_dict["abstract"]-debug_dict["explicit"])*(debug_dict["abstract"]-debug_dict["explicit"])).sum()
#print("H_diff {}".format(diff_term))
from explicit.genleapfrog_ult_util import getdH
# p_tensor = generate_momentum(q)
# p = Variable(p_tensor,requires_grad=True)
# begin_H = H(q,p,alpha)
# print("begin H {}".format(begin_H))
# for i in range(30):
#     #print("i {}".format(i))
#     #print("in q {}".format(q.data))
#     print("H {}".format(H(q, p, alpha)))
#     # dV,H_,dH = getdH(q,V)
#     # lam, Q = eigen(H_.data)
#     # print("lam {}".format(lam))
#     # print("eigen success ")
#     outq, outp = explicit_generalized_leapfrog(q, p, 0.01, alpha, 0.00001, V)
#     # outq_a, outp_a, stat = abstract_generalized_leapfrog(q_point, p_point, 0.1, Ham)
#     q, p = outq, outp
# end_H = H(q, p, alpha)
# print("end_H {}".format(end_H))
# exit()
import math
store = torch.zeros(100,10)
store[0,:] = q.data.clone()
for cur in range(1,100):
    print("cur {}".format(cur))
    p_tensor = generate_momentum(q)
    p = Variable(p_tensor,requires_grad=True)
    begin_H = H(q,p,alpha)
    for i in range(10):
        #print("i {}".format(i))
        #print("in q {}".format(q.data))
        #print("y {}".format(q.data[dim-1]))
        #print("H {}".format(H(q,p,alpha)))
        outq, outp = explicit_generalized_leapfrog(q, p, 0.001, alpha, 0.001, V)
        #outq_a, outp_a, stat = abstract_generalized_leapfrog(q_point, p_point, 0.1, Ham)
        q,p = outq,outp
    end_H = H(q,p,alpha)

    accept_rate = math.exp(min(0,begin_H-end_H))
    if (numpy.random.random(1) < accept_rate):
        pass
    else:
        q_tensor = store[cur-1,:].clone()
        q = Variable(q_tensor,requires_grad=True)
    print("accept_rate {}".format(accept_rate))
    print("begin H {}".format(begin_H))
    print("end H {}".format(end_H))
    print("y {}".format(q.data[dim - 1]))
    #q_point,p_point = outq_a,outp_a
#outq,outp = explicit_generalized_leapfrog(q,p,0.1,alpha,0.1,V)
#outq_a,outp_a,stat = abstract_generalized_leapfrog(q_point,p_point,0.1,Ham)
# compare dV
#print(debug_dict)

#diff_dV = ((debug_dict["abstract"]-debug_dict["explicit"])*(debug_dict["abstract"]-debug_dict["explicit"])).sum()
#print(diff_dV)
#exit()


# diffq = ((outq.data - outq_a.flattened_tensor)*(outq.data - outq_a.flattened_tensor)).sum()
# diffp = ((outp.data - outp_a.flattened_tensor)*(outp.data - outp_a.flattened_tensor)).sum()
# print("diff outq {}".format(diffq))
# print("diff outp {}".format(diffp))
# print("end H explicit{}".format(H(q,p,alpha)))
# print("end H abstract {}".format(Ham.evaluate(q_point,p_point)))

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