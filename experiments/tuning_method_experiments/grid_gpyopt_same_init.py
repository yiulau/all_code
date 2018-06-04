import torch
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from abstract.abstract_class_Ham import Hamiltonian
from abstract.metric import metric
from abstract.abstract_class_point import point

ep_bounds = (0.001,0.3)
L_bounds = (1,20)

v_obj = V_pima_inidan_logit()

metric_obj = metric("unit_e",v_obj)
Ham = Hamiltonian(v_obj,metric_obj)
init_q_point = point(V=Ham.V)
dim = Ham.V.dim
inputq = torch.randn(dim)
init_q_point.flattened_tensor.copy_(inputq)
init_q_point.load_flatten()

num_opt_updates = 20
num_reps_per_update = 100






