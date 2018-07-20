import dill as pickle
from abstract.util import wrap_V_class_with_input_data
from distributions.neural_nets.priors.prior_util import prior_generator
import os, numpy,torch
import dill as pickle
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from post_processing.ESS_nuts import ess_stan,diagnostics_stan
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics
from input_data.convert_data_to_dict import get_data_dict
from distributions.neural_nets.fc_V_model_4 import V_fc_model_4
from post_processing.test_error import test_error

input_data = get_data_dict("8x8mnist")
input_data = get_data_dict("8x8mnist",standardize_predictor=True)
test_set = {"input":input_data["input"][-500:,],"target":input_data["target"][-500:]}
train_set = {"input":input_data["input"][:500,],"target":input_data["target"][:500]}

model_dict = {"num_units": 35}

prior_dict = {"name": "normal"}

V_fun = wrap_V_class_with_input_data(class_constructor=V_fc_model_4, input_data=train_set,
                                                 prior_dict=prior_dict, model_dict=model_dict)
store_name = "normal_35_xhmc_sampler.pkl"

sampler1 = pickle.load(open(store_name, 'rb'))


mcmc_samples_hidden_in = sampler1.get_samples_alt(prior_obj_name="hidden_in",permuted=False)
mcmc_samples_hidden_out = sampler1.get_samples_alt(prior_obj_name="hidden_out",permuted=False)

#print(mcmc_samples_beta["indices_dict"])
#exit()

samples = mcmc_samples_hidden_in["samples"]
#hidden_in_tau_indices = mcmc_samples_hidden_in["indices_dict"]["tau"]
#hidden_in_c_indices = mcmc_samples_hidden_in["indices_dict"]["c"]
#hidden_in_lamb_indices = mcmc_samples_hidden_in["indices_dict"]["lamb"]
#hidden_in_lamb_tilde_indices = mcmc_samples_hidden_in["indices_dict"]["lamb_tilde"]

#hidden_in_sigma2_indices = mcmc_samples_hidden_in["indices_dict"]["sigma2"]
hidden_in_w_indices = mcmc_samples_hidden_in["indices_dict"]["w"]
hidden_out_w_indices = mcmc_samples_hidden_out["indices_dict"]["w"]

# print(samples[:,:,hidden_in_tau_indices].shape)
# print(numpy.mean(samples[3,400:500,hidden_in_tau_indices]))
#
# #print(samples.shape)
#posterior_mean_hidden_in_tau = numpy.mean(samples[:,:,hidden_in_tau_indices].reshape(-1,len(hidden_in_tau_indices)),axis=0)
#
#posterior_mean_hidden_in_c = numpy.mean(samples[:,:,hidden_in_c_indices].reshape(-1,len(hidden_in_c_indices)),axis=0)
#
# print("diagnostics for tau")
# #
# print(diagnostics_stan(samples[:,:,hidden_in_tau_indices]))
# #
# print("posterior mean tau {}".format(posterior_mean_hidden_in_tau))



# print("diagnostics for sigma2")
# print(diagnostics_stan(samples[:,:,hidden_in_sigma2_indices]))
#
# posterior_mean_hidden_in_sigma2 = numpy.mean(samples[:,:,hidden_in_sigma2_indices].reshape(-1,len(hidden_in_sigma2_indices)),axis=0)
# #print("diagnostics for lamb")
#
# print(diagnostics_stan(samples[:,:,hidden_in_lamb_indices]))
#
#
# print("diagnostics for lamb tilde ")
# print(diagnostics_stan(samples[:,:,hidden_in_lamb_tilde_indices]))
#
#
# print("diagnostics for c")
# #
# print(diagnostics_stan(samples[:,:,hidden_in_c_indices]))
# #
# print("posterior mean c {}".format(posterior_mean_hidden_in_c))

print("overall diagnostics")
full_mcmc_tensor = get_params_mcmc_tensor(sampler=sampler1)

print(get_short_diagnostics(full_mcmc_tensor))

#print(mcmc_samples_beta["indices_dict"])

out = sampler1.get_diagnostics(permuted=False)

print("num divergences after warmup")
processed_diag = process_diagnostics(out,name_list=["divergent"])

print(processed_diag.sum(axis=1))

print("num hit max tree depth after warmup")
processed_diag = process_diagnostics(out,name_list=["hit_max_tree_depth"])

print(processed_diag.sum(axis=1))

print("average number of leapfrog steps after warmup")
processed_diag = process_diagnostics(out,name_list=["num_transitions"])
print(processed_diag.mean(axis=1))

#processed_energy = process_diagnostics(out,name_list=["prop_H"])

print("energy diagnostics")
print(energy_diagnostics(diagnostics_obj=out))

mcmc_samples_mixed = sampler1.get_samples(permuted=True)

v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_model_4,input_data=train_set,prior_dict=prior_dict,model_dict=model_dict)
precision_type = "torch.DoubleTensor"

te1,predicted1 = test_error(test_set,v_obj=v_generator(precision_type=precision_type),mcmc_samples=mcmc_samples_mixed,type="classification",memory_efficient=False)

print(te1)


exit()


prior_names_list = ['horseshoe_3',"horseshoe_ard_2","horseshoe_ard","normal","rhorseshoe_3","rhorseshoe_ard_2",
                    "rhorseshoe_ard","gaussian_inv_gamma_2","gaussian_inv_gamma_ard_2","gaussian_inv_gamma_ard"]

num_units_list = [15,35,55]

methods_list = ["gnuts","xhmc"]

for i in range(len(prior_names_list)):
    for j in range(len(num_units_list)):
        for k in range(len(methods_list)):
            store_name = "{}_{}_{}_sampler.pkl".format(prior_names_list[i],num_units_list[j],methods_list[k])