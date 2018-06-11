from distributions.linear_regressions.linear_regression import V_linear_regression

import numpy
import pickle
import torch
import os
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from distributions.two_d_normal import V_2dnormal
from experiments.correctdist_experiments.prototype import check_mean_var_stan

from post_processing.ESS_nuts import ess_stan

v_obj = V_linear_regression()
print(v_obj.X.shape)
print(v_obj.y.shape)


seedid = 33
numpy.random.seed(seedid)
torch.manual_seed(seedid)
mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False)

# input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
#               "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict = {"v_fun":[V_linear_regression],"epsilon":["dual"],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[True],"windowed":[False],"criterion":[None]}
input_dict = {"v_fun":[V_linear_regression],"epsilon":["dual"],"second_order":[False],
              "metric_name":["unit_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}
input_dict = {"v_fun":[V_linear_regression],"epsilon":["dual"],"second_order":[False],"cov":["adapt"],
              "metric_name":["dense_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}

ep_dual_metadata_argument = {"name":"epsilon","target":0.8,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
dual_args_list = [ep_dual_metadata_argument]

other_arguments = other_default_arguments()
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=13)]

tune_settings_dict = tuning_settings(dual_args_list,[],adapt_cov_arguments,other_arguments)

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()
#tune_settings_dict = tuning_settings([],[],[],[])

#tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

out = sampler1.start_sampling()

sampler1.remove_failed_chains()

print("num chains removed {}".format(sampler1.metadata.num_chains_removed))
print("num restarts {}".format(sampler1.metadata.num_restarts))
mcmc_samples = sampler1.get_samples(permuted=True)
with open("debug_test_error_mcmc_regression.pkl", 'wb') as f:
    pickle.dump(mcmc_samples, f)
exit()

