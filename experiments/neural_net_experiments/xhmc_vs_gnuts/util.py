from distributions.neural_nets.priors.prior_util import prior_generator
import os, numpy,torch
from abstract.util import wrap_V_class_with_input_data

import dill as pickle
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from post_processing.ESS_nuts import ess_stan,diagnostics_stan
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics
from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import test_error
def run_nn_experiment(xhmc_delta_list,input_data,v_fun,test_set,type_problem):
    out_list = [None]*(len(xhmc_delta_list)+1)
    for i in range(len(out_list)):
        model_dict = {"num_units": 50}

        v_generator = wrap_V_class_with_input_data(class_constructor=v_fun, input_data=input_data,
                                                   model_dict=model_dict)

        mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=2000, num_chains=4, num_cpu=4, thin=1,
                                               tune_l_per_chain=1000,
                                               warmup_per_chain=1100, is_float=False, isstore_to_disk=False,
                                               allow_restart=False)


        if i<len(out_list)-1:
            input_dict = {"v_fun": [v_generator], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
                          "max_tree_depth": [8],"xhmc_delta":[xhmc_delta_list[i]],
                          "metric_name": ["diag_e"], "dynamic": [True], "windowed": [False], "criterion": ["xhmc"]}
        else:
            input_dict = {"v_fun": [v_generator], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
                          "max_tree_depth": [8],
                          "metric_name": ["diag_e"], "dynamic": [True], "windowed": [False], "criterion": ["gnuts"]}

        ep_dual_metadata_argument = {"name": "epsilon", "target": 0.9, "gamma": 0.05, "t_0": 10,
                                     "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}
        #
        adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=v_generator(
            precision_type="torch.DoubleTensor").get_model_dim())]
        dual_args_list = [ep_dual_metadata_argument]
        other_arguments = other_default_arguments()
        # tune_settings_dict = tuning_settings([],[],[],[])
        tune_settings_dict = tuning_settings(dual_args_list, [], adapt_cov_arguments, other_arguments)
        tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

        sampler1 = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta,
                                tune_settings_dict=tune_settings_dict)


        diagnostics_np = sampler1.np_diagnostics()
        samples_mixed = sampler1.get_samples(permuted=True)
        te = test_error(target_dataset=test_set,v_obj=v_generator("torch.DoubleTensor"),mcmc_samples=samples_mixed,type=type_problem)

        out = {"test_error":te,"diagnostics":diagnostics_np}
        out_list.append(out)


    te_store = numpy.zeros(len(out_list))
    diagnostics_store = numpy.zeros(shape=[len(out_list)]+list(diagnostics_np.shape))
    for i in range(len(out_list)):
        te_store[i] = out_list[i]["test_error"]
        diagnostics_store[i,...] = out_list[i]["diagnostics"]

    output_store = {"test_error":te_store,"diagnostics":diagnostics_store}
    save_name = "xhmc_v_gnuts_8x8mnist.npz"
    numpy.savez(save_name,**output_store)
    return(save_name)