import numpy
import pickle
import torch
import os
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from input_data.convert_data_to_dict import get_data_dict
from abstract.util import wrap_V_class_with_input_data
# seedid = 30
# numpy.random.seed(seedid)
# torch.manual_seed(seedid)
mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=500,
                                   warmup_per_chain=600,is_float=False,isstore_to_disk=False,allow_restart=False)
input_data = get_data_dict("pima_indian")
V_pima_indian_logit = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

input_dict = {"v_fun":[V_pima_indian_logit],"epsilon":["dual"],"second_order":[False],"cov":["adapt"],
              "metric_name":["dense_e"],"dynamic":[True],"windowed":[False],"criterion":["nuts"]}

ep_dual_metadata_argument = {"name":"epsilon","target":0.8,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=7)]
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()

tune_settings_dict = tuning_settings(dual_args_list,[],adapt_cov_arguments,other_arguments)

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()


sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

out = sampler1.start_sampling()



mcmc_samples = sampler1.get_samples(permuted=False)
print("mcmc mean {}".format(numpy.mean(mcmc_samples,axis=0)))
#print(numpy.cov(mcmc_samples,rowvar=False))

address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain.pkl"
correct = pickle.load(open(address, 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()
print("exact mean {}".format(correct_mean))
#print(correct_cov)

output = check_mean_var_stan(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = output["mcmc_mean"],output["mcmc_Cov"]
pc_mean,pc_cov = output["pc_of_mean"],output["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)