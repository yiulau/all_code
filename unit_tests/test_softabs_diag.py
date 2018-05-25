# check that abstract leapfrog and explicit leapfrog gives the same answer
import numpy
import pandas as pd
import torch
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_genleapfrog_ult_util import generalized_leapfrog_softabsdiag
from abstract.metric import metric
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from torch.autograd import Variable
from abstract.abstract_class_point import point
from explicit.genleapfrog_ult_util import getH, eigen, softabs_map
import os

seedid = 33
numpy.random.seed(seedid)
torch.manual_seed(seedid)

alpha = 1e6
#debug_dict = {"abstract":None,"explicit":None}

#debug_dict.update({"explicit":y.data.clone()})
# first verify they have the same Hamiltonian function
inputq = torch.randn(7)

v_obj = V_pima_inidan_logit()
metric_obj = metric("softabs_diag",v_obj,alpha)
Ham = Hamiltonian(v_obj,metric_obj)
q_point = point(V=Ham.V)


q_point.flattened_tensor.copy_(inputq)
q_point.load_flatten()
p_point = Ham.T.generate_momentum(q_point)

print("abstract H {}".format(Ham.evaluate(q_point,p_point)))
print("abstract V {}".format(Ham.V.evaluate_scalar(q_point)))
print("abstract T {}".format(Ham.T.evaluate_scalar(q_point,p_point)))


L= 1
for i in range(L):
    outq_a, outp_a, stat = generalized_leapfrog_softabsdiag(q_point, p_point, 0.1, Ham)
    q_point,p_point = outq_a,outp_a

print("end H abstract {}".format(Ham.evaluate(q_point,p_point)))
print(stat.divergent)