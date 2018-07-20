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

prior_names_list = ["normal","rhorseshoe_ard","horseshoe_ard","rhorseshoe_3","horseshoe_3",
                    "gaussian_inv_gamma_ard","gaussian_inv_gamma_2"]

num_units_list = [35]

methods_list = ["xhmc"]

for i in range(len(prior_names_list)):
    for j in range(len(num_units_list)):
        for k in range(len(methods_list)):
            method = methods_list[k]
            input_data = get_data_dict("8x8mnist")
            input_data = {"input": input_data["input"][:500, ], "target": input_data["target"][:500]}
            model_dict = {"num_units": num_units_list[j]}

            prior_dict = {"name": prior_names_list[i]}


            V_fun = wrap_V_class_with_input_data(class_constructor=V_fc_model_4, input_data=input_data,
                                                 prior_dict=prior_dict, model_dict=model_dict)

            mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=2000, num_chains=4, num_cpu=4, thin=1,
                                                   tune_l_per_chain=1000,
                                                   warmup_per_chain=1100, is_float=False, isstore_to_disk=False,
                                                   allow_restart=False,seed=14)

            if method=="xhmc":
                input_dict = {"v_fun": [V_fun], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
                              "max_tree_depth": [8], "xhmc_delta": [0.1],
                              "metric_name": ["diag_e"], "dynamic": [True], "windowed": [False], "criterion": ["xhmc"]}
            else:
                input_dict = {"v_fun": [V_fun], "epsilon": ["dual"], "second_order": [False],"max_tree_depth": [8],
                              "metric_name": ["diag_e"], "dynamic": [True], "windowed": [False], "criterion": ["gnuts"]}

            ep_dual_metadata_argument = {"name": "epsilon", "target": 0.85, "gamma": 0.05, "t_0": 10,
                                         "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}
            #
            adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=V_fun(
                precision_type="torch.DoubleTensor").get_model_dim())]
            dual_args_list = [ep_dual_metadata_argument]
            other_arguments = other_default_arguments()
            tune_settings_dict = tuning_settings(dual_args_list, [], adapt_cov_arguments, other_arguments)
            tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

            sampler1 = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta,
                                    tune_settings_dict=tune_settings_dict)


            store_name = "{}_{}_{}_sampler.pkl".format(prior_names_list[i],num_units_list[j],methods_list[k])
            sampled = False
            if sampled:
                sampler1 = pickle.load(open(store_name, 'rb'))
            else:
                sampler1.start_sampling()
                with open(store_name, 'wb') as f:
                    pickle.dump(sampler1, f)


