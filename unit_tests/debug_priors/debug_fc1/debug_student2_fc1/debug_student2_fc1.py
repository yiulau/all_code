# perfect separation logit data

from distributions.neural_nets.fc_V_model_1 import V_fc_model_1
from abstract.util import wrap_V_class_with_input_data
from distributions.neural_nets.priors.prior_util import prior_generator
import os, numpy,torch
import dill as pickle
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from post_processing.ESS_nuts import ess_stan,diagnostics_stan
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics
from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import test_error
seed = 1
numpy.random.seed(seed)
torch.manual_seed(seed)

input_data = get_data_dict("pima_indian",standardize_predictor=True)


prior_dict = {"name":"gaussian_inv_gamma_2"}
model_dict = {"num_units":10}

v_generator =wrap_V_class_with_input_data(class_constructor=V_fc_model_1,input_data=input_data,prior_dict=prior_dict,model_dict=model_dict)

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=4,num_cpu=4,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False,allow_restart=False)

# input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
#                "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict = {"v_fun":[v_generator],"epsilon":["dual"],"second_order":[False],"cov":["adapt"],"max_tree_depth":[8],
               "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}
# input_dict = {"v_fun":[v_generator],"epsilon":[0.1],"second_order":[False],"evolve_L":[10],
#               "metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}
ep_dual_metadata_argument = {"name":"epsilon","target":0.9,"gamma":0.05,"t_0":10,
                         "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
#
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=v_generator(precision_type="torch.DoubleTensor").get_model_dim())]
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()
#tune_settings_dict = tuning_settings([],[],[],[])
tune_settings_dict = tuning_settings(dual_args_list,[],adapt_cov_arguments,other_arguments)
tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)


store_name = 'student2_fc1_sampler.pkl'
sampled = True
if sampled:
    sampler1 = pickle.load(open(store_name, 'rb'))
else:
    sampler1.start_sampling()
    with open(store_name, 'wb') as f:
        pickle.dump(sampler1, f)
#out = sampler1.start_sampling()


mcmc_samples_hidden_in = sampler1.get_samples_alt(prior_obj_name="hidden_in",permuted=False)
print(mcmc_samples_hidden_in["samples"].shape)

print(mcmc_samples_hidden_in["samples"][0,10,5])
print(mcmc_samples_hidden_in["samples"][1,10,5])
#exit()
mcmc_samples_hidden_out = sampler1.get_samples_alt(prior_obj_name="hidden_out",permuted=False)

#print(mcmc_samples_beta["indices_dict"])
#exit()

samples = mcmc_samples_hidden_in["samples"]
hidden_in_sigma2_indices = mcmc_samples_hidden_in["indices_dict"]["sigma2"]
hidden_in_w_indices = mcmc_samples_hidden_in["indices_dict"]["w"]
hidden_out_w_indices = mcmc_samples_hidden_out["indices_dict"]["w"]
#print(samples.shape)
posterior_mean_hidden_in_sigma2 = numpy.mean(samples[:,:,hidden_in_sigma2_indices].reshape(-1,len(hidden_in_sigma2_indices)),axis=0)

print(diagnostics_stan(samples[:,:,hidden_in_sigma2_indices]))

print("hidden in sigma2 {}".format(posterior_mean_hidden_in_sigma2))

#print(mcmc_samples_beta["indices_dict"])

out = sampler1.get_diagnostics(permuted=False)


processed_diag = process_diagnostics(out,name_list=["divergent"])

print(processed_diag.sum(axis=1))
exit()
#print(processed_diag.shape)

#processed_energy = process_diagnostics(out,name_list=["prop_H"])

print(energy_diagnostics(diagnostics_obj=out))

mcmc_samples_mixed = sampler1.get_samples(permuted=True)
precision_type = "torch.DoubleTensor"
te2,predicted2 = test_error(input_data,v_obj=v_generator(precision_type=precision_type),mcmc_samples=mcmc_samples_mixed,type="classification",memory_efficient=False)

print(te2)