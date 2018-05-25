import numpy
import pandas as pd
import torch
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_leapfrog_ult_util import abstract_leapfrog_ult
from abstract.metric import metric
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from explicit.general_util import logsumexp_torch
from torch.autograd import Variable
from abstract.abstract_class_point import point
from explicit.leapfrog_ult_util import leapfrog_ult
#y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
#X_np = numpy.random.randn(num_ob,dim)
seedid = 33
numpy.random.seed(seedid)
torch.manual_seed(seedid)
import os

inputq = torch.randn(7)
inputp = torch.randn(7)

v_obj = V_pima_inidan_logit()
metric_obj = metric("unit_e",v_obj)
Ham = Hamiltonian(v_obj,metric_obj)
q_point = point(V=Ham.V)
p_point = point(T=Ham.T)

q_point.flattened_tensor.copy_(inputq)
p_point.flattened_tensor.copy_(inputp)
q_point.load_flatten()
p_point.load_flatten()

print("abstract H {}".format(Ham.evaluate(q_point,p_point)))
print("input q {}".format(q_point.flattened_tensor))
print("input p {}".format(p_point.flattened_tensor))

outq_a,outp_a,stat = abstract_leapfrog_ult(q_point,p_point,0.1,Ham)
print("output q {} ".format(outq_a.flattened_tensor))
print("output p {}".format(outp_a.flattened_tensor))

outp_a.flattened_tensor *= -1
outp_a.load_flatten()
backq, backp, stat = abstract_leapfrog_ult(outq_a,outp_a,0.1,Ham)
print("back q {}".format(backq.flattened_tensor))
print("back p {}".format(backp.flattened_tensor))
exit()
L=10
for i in range(L):
    outq,outp = leapfrog_ult(q,p,0.1,H)
    outq_a,outp_a,stat = abstract_leapfrog_ult(q_point,p_point,0.1,Ham)
    q, p = outq, outp
    q_point, p_point = outq_a, outp_a
diffq = ((outq.data - outq_a.flattened_tensor)*(outq.data - outq_a.flattened_tensor)).sum()
diffp = ((outp.data - outp_a.flattened_tensor)*(outp.data - outp_a.flattened_tensor)).sum()
print("diff outq {}".format(diffq))
print("diff outp {}".format(diffp))
