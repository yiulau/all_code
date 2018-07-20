import numpy
import pickle
import torch
import os
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.neural_nets.fc_V_model_debug import V_fc_model_debug
from experiments.experiment_obj import tuneinput_class
from distributions.two_d_normal import V_2dnormal
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from post_processing.ESS_nuts import ess_stan
from abstract.abstract_class_point import point
from input_data.convert_data_to_dict import get_data_dict

seedid = 30
numpy.random.seed(seedid)
torch.manual_seed(seedid)


model_dict = {"num_units":20}
input_data = get_data_dict("8x8mnist")
v_obj = V_fc_model_debug(precision_type="torch.DoubleTensor",model_dict=model_dict,input_data=input_data)

q = point(V=v_obj)

print(q.flattened_tensor)


q_clone = q.point_clone()

print(q_clone.flattened_tensor)

