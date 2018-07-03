from distributions.test_hierarchical_priors.V_logit.V_logit import V_logit
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
seed = 1
numpy.random.seed(seed)
n =30
dim= 100
X = numpy.random.randn(n,dim)
y = numpy.zeros(n)

for i in range(n):
    y[i] = numpy.asscalar(numpy.random.choice(2,1))
    if y[i] > 0:
        X[i,0:2] = numpy.random.randn(2)*0.5 + 1
    else:
        X[i,0:2] = numpy.random.randn(2)*0.5 -1

input_data = {"target":y,"input":X}


prior_dict = {"name":"horseshoe_3"}

v_generator =wrap_V_class_with_input_data(class_constructor=V_logit,input_data=input_data,prior_dict=prior_dict)

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=1000,
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


store_name = 'hs3_logit_sampler.pkl'
sampled = True
if sampled:
    sampler1 = pickle.load(open(store_name, 'rb'))
else:
    sampler1.start_sampling()
    with open(store_name, 'wb') as f:
        pickle.dump(sampler1, f)
#out = sampler1.start_sampling()


mcmc_samples_beta = sampler1.get_samples_alt(prior_obj_name="beta",permuted=False)
#print(mcmc_samples_beta["indices_dict"])
#exit()

samples = mcmc_samples_beta["samples"]
w_indices = mcmc_samples_beta["indices_dict"]["w"]
tau_indices = mcmc_samples_beta["indices_dict"]["tau"]

print(samples.shape)
posterior_mean = numpy.mean(samples[:,:,w_indices].reshape(-1,len(w_indices)),axis=0)
print(posterior_mean[:2])

posterior_mean_tau = numpy.mean(samples[:,:,tau_indices].reshape(-1,len(tau_indices)),axis=0)

print(diagnostics_stan(samples[:,:,tau_indices]))

print("posterior mean tau {}".format(posterior_mean_tau))

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
mixed_mcmc_tensor = sampler1.get_samples(permuted=True)
print(mixed_mcmc_tensor)

mcmc_cov = numpy.cov(mixed_mcmc_tensor,rowvar=False)
mcmc_sd_vec = numpy.sqrt(numpy.diagonal(mcmc_cov))

print("mcmc problem difficulty")

print(max(mcmc_sd_vec)/min(mcmc_sd_vec))