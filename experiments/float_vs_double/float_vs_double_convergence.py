# mix chains, see if they pass test Gelman-Rubin
import numpy,torch
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from post_processing.get_diagnostics import get_short_diagnostics
from post_processing.ESS_nuts import diagnostics_stan
def convergence_diagnostics(v_fun):
    mcmc_meta_double = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=10000, num_chains=4, num_cpu=1, thin=1,
                                           tune_l_per_chain=1000,
                                           warmup_per_chain=1100, is_float=False, isstore_to_disk=False,
                                           allow_restart=True)
    mcmc_meta_float = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=10000, num_chains=4, num_cpu=1, thin=1,
                                                  tune_l_per_chain=1000,
                                                  warmup_per_chain=1100, is_float=True, isstore_to_disk=False,
                                                  allow_restart=True)

    input_dict = {"v_fun": [v_fun], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
                  "metric_name": ["dense_e"], "dynamic": [False], "windowed": [False],
                  "criterion": ["gnuts"]}
    ep_dual_metadata_argument = {"name": "epsilon", "target": 0.65, "gamma": 0.05, "t_0": 10,
                                 "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}

    dim = len(v_fun().flattened_tensor)
    adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=dim)]
    dual_args_list = [ep_dual_metadata_argument]
    other_arguments = other_default_arguments()

    tune_settings_dict = tuning_settings(dual_args_list, [], adapt_cov_arguments, other_arguments)

    tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

    sampler_double = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta_double, tune_settings_dict=tune_settings_dict)

    sampler_float = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta_float, tune_settings_dict=tune_settings_dict)

    sampler_float.start_sampling()

    sampler_double.start_sampling()

    sampler_double.remove_failed_chains()

    sampler_float.remove_failed_chains()

    output_float = sampler_float.get_samples_p_diag(permuted=False)
    output_double = sampler_double.get_samples_p_diag(permuted=False)
    short_diagnostics_float = get_short_diagnostics(output_float)
    short_diagnostics_double = get_short_diagnostics(output_double)



    samples_double_cast_to_float = output_double["samples"].type("torch.FloatTensor")
    samples_float = output_float["samples"]

    combined_samples = torch.cat([samples_double_cast_to_float,samples_float],dim=0)

    short_diagnostics_combined = get_short_diagnostics(combined_samples)

    out = {"diag_combined":short_diagnostics_combined,"diag_float":short_diagnostics_float,"diag_double":short_diagnostics_double}
    return(out)



