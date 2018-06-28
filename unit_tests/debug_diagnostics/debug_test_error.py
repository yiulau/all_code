import numpy
import dill as pickle
import torch
import os
from post_processing.test_error import map_prediction,test_error
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class
from distributions.two_d_normal import V_2dnormal
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from abstract.util import wrap_V_class_with_input_data
from post_processing.ESS_nuts import ess_stan
from input_data.convert_data_to_dict import get_data_dict
seedid = 30
numpy.random.seed(seedid)
torch.manual_seed(seedid)

input_data = get_data_dict("pima_indian")
v_generator = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=1000,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=200,is_float=False,isstore_to_disk=False,allow_restart=True)

# input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
#               "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict = {"v_fun":[v_generator],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

tune_settings_dict = tuning_settings([],[],[],[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

sampled = True
address = "debug_test_error_mcmc.pkl"
if sampled:
    with open(address, 'rb') as f:
        sampler1 = pickle.load(f)

else:
    sampler1.start_sampling()
    with open(address, 'wb') as f:
        pickle.dump(sampler1, f)


sampler1.remove_failed_chains()
print("num chains removed {}".format(sampler1.metadata.num_chains_removed))
print("num restarts {}".format(sampler1.metadata.num_restarts))

mcmc_samples = sampler1.get_samples(permuted=False)

out = sampler1.get_samples_p_diag(permuted=False)



#print(mcmc_samples.shape)
#print("mcmc mean {}".format(numpy.mean(mcmc_samples,axis=0)))
print(ess_stan(mcmc_samples))
#from python2R.ess_rpy2 import ess_repy2
mcmc_samples_mixed = sampler1.get_samples(permuted=True)
target_dataset = get_data_dict("pima_indian")


v_generator = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)
precision_type = "torch.DoubleTensor"

te1,predicted1 = test_error(target_dataset,v_obj=v_generator(precision_type=precision_type),mcmc_samples=mcmc_samples_mixed,type="classification",memory_efficient=False)
te2,predicted2 = test_error(target_dataset,v_obj=v_generator(precision_type=precision_type),mcmc_samples=mcmc_samples_mixed,type="classification",memory_efficient=True)

print(te1)
print(te2)