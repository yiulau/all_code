import numpy
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from distributions.response_model import V_response_model
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=5000,num_chains=4,num_cpu=4,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False,allow_restart=False)

input_data = get_data_dict("1-PL",standardize_predictor=False)
V_generator = wrap_V_class_with_input_data(class_constructor=V_response_model,input_data=input_data)

input_dict = {"v_fun":[V_generator],"epsilon":["dual"],"second_order":[False],"cov":["adapt"],
              "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}

ep_dual_metadata_argument = {"name":"epsilon","target":0.95,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=V_generator(precision_type="torch.DoubleTensor").get_model_dim())]
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()
tune_settings_dict = tuning_settings(dual_args_list,[],adapt_cov_arguments,other_arguments)
tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()
sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)
sampler1.start_sampling()

print("overall diagnostics")
full_mcmc_tensor = sampler1.get_samples(permuted=False)

print(get_short_diagnostics(full_mcmc_tensor))

out = sampler1.get_diagnostics(permuted=False)

print("average acceptance rate after warmup")
processed_diag = process_diagnostics(out,name_list=["accept_rate"])

average_accept_rate = numpy.mean(processed_diag,axis=1)

print(average_accept_rate)

print("energy diagnostics")
print(energy_diagnostics(diagnostics_obj=out))


mixed_mcmc_tensor = sampler1.get_samples(permuted=True)
print(mixed_mcmc_tensor)

true_cov = numpy.cov(mixed_mcmc_tensor,rowvar=False)
sd_vec = numpy.diagonal(true_cov)

print("problem difficulty")

print(max(sd_vec)/min(sd_vec)) # val = 11.5