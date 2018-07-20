from distributions.mvn import V_mvn
from input_data.convert_data_to_dict import get_data_dict
from abstract.util import wrap_V_class_with_input_data
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from experiments.correctdist_experiments.result_from_long_chain.logistic.util import result_from_long_chain
import numpy

from experiments.experiment_util import wishart_for_cov
import os

dim = 100
input_data = {"input":wishart_for_cov(dim=dim,seed=dim)}
v_generator = wrap_V_class_with_input_data(class_constructor=V_mvn, input_data=input_data)

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=1000, num_chains=4, num_cpu=4, thin=1,
                                       tune_l_per_chain=1000,
                                       warmup_per_chain=1100, is_float=False, isstore_to_disk=False,
                                       allow_restart=False)


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
tune_settings_dict = tuning_settings(dual_args_list, [], adapt_cov_arguments, other_arguments)
tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta, tune_settings_dict=tune_settings_dict)

sampler1.start_sampling()

mcmc_samples = sampler1.get_samples(permuted=False)

correct_mean = numpy.zeros(dim)
correct_cov = numpy.linalg.inv(input_data["input"])
check_result= check_mean_var_stan(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov)
print(check_result)
save_name = "check_result_{}.npz".format("mvn_{}".format(100))
numpy.savez(save_name,**check_result)