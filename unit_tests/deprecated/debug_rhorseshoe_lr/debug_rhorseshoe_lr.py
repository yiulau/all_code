from distributions.test_hierarchical_priors.rhorseshoe_lr import V_rhs_lr
from abstract.util import wrap_V_class_with_input_data
from distributions.neural_nets.priors.prior_util import prior_generator
import os, numpy,torch
import dill as pickle
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from post_processing.ESS_nuts import ess_stan
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics

seed = 1
numpy.random.seed(seed)
non_zero_num_p = 20
full_p = 400
num_samples = 100
X_np = numpy.random.randn(num_samples,full_p)*5
true_beta = numpy.zeros(full_p)
true_beta[:non_zero_num_p] = numpy.random.randn(non_zero_num_p)*5
y_np = X_np.dot(true_beta) + numpy.random.randn(num_samples)
input_data = {"input":X_np,"target":y_np}


v_generator =wrap_V_class_with_input_data(class_constructor=V_rhs_lr,input_data=input_data)

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=2,num_cpu=1,thin=1,tune_l_per_chain=1000,
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


store_name = 'rhs_lr_sampler.pkl'
sampled = False
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
print(samples.shape)
posterior_mean = numpy.mean(samples[:,:,w_indices].reshape(-1,len(w_indices)),axis=0)
print(posterior_mean[:non_zero_num_p])
print(true_beta[:non_zero_num_p])

#print(mcmc_samples_beta["indices_dict"])

out = sampler1.get_diagnostics(permuted=False)


#processed_diag = process_diagnostics(out,name_list=["accepted"])
#print(processed_diag.shape)

#processed_energy = process_diagnostics(out,name_list=["prop_H"])

print(energy_diagnostics(diagnostics_obj=out))