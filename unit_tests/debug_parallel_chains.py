import pickle

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from experiments.experiment_obj import tuneinput_class

from experiments.correctdist_experiments.prototype import check_mean_var

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=10,num_chains=2,num_cpu=2,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=5,is_float=False,isstore_to_disk=False)


input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

tune_settings_dict = tuning_settings([],[],[],[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)


import time
start_time = time.time()
out = sampler1.start_sampling()

print("total time {}".format(time.time()-start_time))
