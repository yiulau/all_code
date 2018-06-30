import pickle,os

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict

input_data = get_data_dict("pima_indian")
V_pima_indian_logit = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2500,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_pima_indian_logit],"epsilon":[0.1],"second_order":[False],"cov":["adapt"],
              "evolve_L":[10],"metric_name":["diag_e"],"dynamic":[False],"windowed":[False]}

adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=7)]

tune_settings_dict = tuning_settings([],[],adapt_cov_arguments,[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()


sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)



out = sampler1.start_sampling()
