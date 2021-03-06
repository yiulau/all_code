import numpy
import dill as pickle
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from distributions.mvn import V_mvn
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics
from experiments.experiment_util import wishart_for_cov
mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=4,num_cpu=4,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False,allow_restart=False)

numpy.random.seed(0)
Sigma_inv = wishart_for_cov(dim=50)
Sigma = numpy.linalg.inv(Sigma_inv)
# sd_vec = numpy.sqrt(numpy.diagonal(Sigma))
# print(max(sd_vec))
# print(min(sd_vec))
# exit()
input_data = {"input":Sigma_inv}


V_generator = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)

input_dict = {"v_fun":[V_generator],"epsilon":["dual"],"second_order":[False],"cov":["adapt"],
              "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}

ep_dual_metadata_argument = {"name":"epsilon","target":0.65,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=V_generator(precision_type="torch.DoubleTensor").get_model_dim())]
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()
tune_settings_dict = tuning_settings(dual_args_list,[],adapt_cov_arguments,other_arguments)
tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()
sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

store_name = 'mvn_sampler.pkl'
sampled = True
if sampled:
    sampler1 = pickle.load(open(store_name, 'rb'))
else:
    sampler1.start_sampling()
    with open(store_name, 'wb') as f:
        pickle.dump(sampler1, f)

print("overall diagnostics")
full_mcmc_tensor = sampler1.get_samples(permuted=False)

print(get_short_diagnostics(full_mcmc_tensor))

out = sampler1.get_diagnostics(permuted=False)
print("num divergent")
processed_diag = process_diagnostics(out,name_list=["divergent"])

print(processed_diag.sum(axis=1))
print("num hit max tree depth")
processed_diag = process_diagnostics(out,name_list=["hit_max_tree_depth"])

print(processed_diag.sum(axis=1))

print("average acceptance rate after warmup")
processed_diag = process_diagnostics(out,name_list=["accept_rate"])

average_accept_rate = numpy.mean(processed_diag,axis=1)

print(average_accept_rate)

print("energy diagnostics")
print(energy_diagnostics(diagnostics_obj=out))


mixed_mcmc_tensor = sampler1.get_samples(permuted=True)
print(mixed_mcmc_tensor)

mcmc_cov = numpy.cov(mixed_mcmc_tensor,rowvar=False)
mcmc_sd_vec = numpy.sqrt(numpy.diagonal(mcmc_cov))
print("mcmc sd vec")
print(mcmc_sd_vec)
print("mcmc problem difficulty")

print(max(mcmc_sd_vec)/min(mcmc_sd_vec)) # val should be ~ 1.5 get 4.5

sd_vec = numpy.sqrt(numpy.diagonal(Sigma))
print("true sd vec")
print(sd_vec)
print("true difficulty")
print(max(sd_vec)/min(sd_vec))
exit()