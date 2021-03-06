
from distributions.funnel_cp import V_funnel_cp
from distributions.funnel_ncp import V_funnel_ncp
import numpy
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.experiment_obj import experiment,experiment_setting_dict
import torch
#from experiments.correctdist_experiments.prototype import check_mean_var
seedid = 2
numpy.random.seed(seedid)
torch.manual_seed(seedid)
precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)
mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=100,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=10,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_funnel_cp],"epsilon":[0.1],"alpha":[1e6],"second_order":[True],
              "evolve_L":[10],"metric_name":["softabs"],"dynamic":[False],"windowed":[False],"criterion":[None]}

tune_settings_dict = tuning_settings([],[],[],[])
tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)




out = sampler1.start_sampling()
mcmc_samples = sampler1.get_samples(permuted=True)

print("mcmc mean {}".format(numpy.mean(mcmc_samples,axis=0)))
print("mcmc cov {}".format(numpy.cov(mcmc_samples,rowvar=False)))
#print(mcmc_samples)