import numpy
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from distributions.stochastic_volatility.stochastic_volatility import V_stochastic_volatility
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=1000,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=100,is_float=False,isstore_to_disk=False,allow_restart=False)

input_data = get_data_dict("sp500")
V_generator = wrap_V_class_with_input_data(class_constructor=V_stochastic_volatility,input_data=input_data)

input_dict = {"v_fun":[V_generator],"epsilon":[0.001],"second_order":[False],
              "metric_name":["unit_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}

other_arguments = other_default_arguments()
tune_settings_dict = tuning_settings([],[],[],other_arguments)
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
