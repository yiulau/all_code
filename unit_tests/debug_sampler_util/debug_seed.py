import numpy,torch
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from post_processing.get_diagnostics import get_short_diagnostics
from post_processing.ESS_nuts import diagnostics_stan
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict
from experiments.float_vs_double.convergence.float_vs_double_convergence import convergence_diagnostics

input_data = get_data_dict("pima_indian")

#input_data = {"input":wishart_for_cov(dim=10)}
v_fun = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)
mcmc_meta_double = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=1500, num_chains=2, num_cpu=2, thin=1,
                                       tune_l_per_chain=1000,
                                       warmup_per_chain=1100, is_float=False, isstore_to_disk=False,
                                       allow_restart=True)
input_dict = {"v_fun": [v_fun], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
              "metric_name": ["diag_e"], "dynamic": [True], "windowed": [False],
              "criterion": ["gnuts"]}
ep_dual_metadata_argument = {"name": "epsilon", "target": 0.8, "gamma": 0.05, "t_0": 10,
                             "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}

dim = len(v_fun(precision_type="torch.DoubleTensor").flattened_tensor)
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=dim)]
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()

tune_settings_dict = tuning_settings(dual_args_list, [], adapt_cov_arguments, other_arguments)

tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

sampler_double = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta_double, tune_settings_dict=tune_settings_dict)

sampler_double.start_sampling()

double_samples = sampler_double.get_samples(permuted=False)

print(double_samples[0,100,2])
print(double_samples[1,100,2])

short_diagnostics_double = get_short_diagnostics(double_samples)

print(short_diagnostics_double)