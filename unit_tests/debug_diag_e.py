import pickle,os

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from experiments.experiment_obj import tuneinput_class

from experiments.correctdist_experiments.prototype import check_mean_var

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=50,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=10,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["diag_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict2 = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["diag_e"],"dynamic":[False],"windowed":[True],"criterion":[None]}


tune_settings_dict = tuning_settings([],[],[],[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

tune_dict2 = tuneinput_class(input_dict2).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)



out = sampler1.start_sampling()