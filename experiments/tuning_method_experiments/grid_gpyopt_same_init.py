import torch,numpy
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from abstract.abstract_class_Ham import Hamiltonian
from abstract.metric import metric
from abstract.abstract_class_point import point
from experiments.experiment_obj import experiment_setting_dict,experiment
from adapt_util.tune_param_classes.tune_param_setting_util import *

ep_bounds = (0.01,0.3)
L_bounds = (1,30)
grid_len = 20

# v_obj = V_pima_inidan_logit()
#
# metric_obj = metric("unit_e",v_obj)
# Ham = Hamiltonian(v_obj,metric_obj)
# init_q_point = point(V=Ham.V)
# dim = Ham.V.dim
# inputq = torch.randn(dim)
# init_q_point.flattened_tensor.copy_(inputq)
# init_q_point.load_flatten()

num_opt_updates = 20
num_reps_per_update = 100
num_samples_per_chain = 5000
warmup_per_sample = 2500
num_chains_per_sampler = 4

# first run grid search

v_fun_list = [V_pima_inidan_logit]
ep_list = list(numpy.linspace(ep_bounds[0],ep_bounds[1],grid_len))
L_list = list(numpy.linspace(L_bounds[0],L_bounds[1],grid_len))
input_dict = {"v_fun":v_fun_list,"epsilon":ep_list,"second_order":[False],
              "evolve_t":L_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}


experiment_setting = experiment_setting_dict(chain_length=num_samples_per_chain,num_repeat=num_opt_updates,num_chains_per_sampler=num_chains_per_sampler)

save_name = "grid_experiment.pkl"
experiment_obj = experiment(input_dict=input_dict,experiment_setting=experiment_setting,save_name=save_name)





