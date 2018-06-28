import pickle

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict

from experiments.correctdist_experiments.prototype import check_mean_var_stan

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=2,num_cpu=1,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False,allow_restart=False)

input_data = get_data_dict("pima_indian")
V_pima_indian_logit = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

# input_dict = {"v_fun":[V_pima_indian_logit],"epsilon":[0.1],"second_order":[False],
#               "evolve_L":[10],"metric_name":["diag_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}
input_dict = {"v_fun":[V_pima_indian_logit],"epsilon":["dual"],"second_order":[False],"cov":["adapt"],
               "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}
ep_dual_metadata_argument = {"name":"epsilon","target":0.9,"gamma":0.05,"t_0":10,
                         "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}

adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=V_pima_indian_logit(precision_type="torch.DoubleTensor").get_model_dim())]
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()
#tune_settings_dict = tuning_settings([],[],[],[])
tune_settings_dict = tuning_settings(dual_args_list,[],adapt_cov_arguments,other_arguments)
#tune_settings_dict = tuning_settings([],[],[],[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)


import time
start_time = time.time()
out = sampler1.parallel_sampling()

print("total time {}".format(time.time()-start_time))
