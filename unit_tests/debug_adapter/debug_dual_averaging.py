import pickle,os

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict

from experiments.correctdist_experiments.prototype import check_mean_var_stan
input_data = get_data_dict("pima_indian")
V_pima_indian_logit = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1000,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_pima_indian_logit],"epsilon":["dual"],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

ep_dual_metadata_argument = {"name":"epsilon","target":0.65,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}

#ep_dual_metadata_argument = dual_default_arguments(name="epsilon")

#evolve_L_opt_metadata_argument = {"name":"evolve_L","obj_fun":"ESJD","bounds":(1,10),"par_type":"medium"}

#alpha_opt_metadata_argument = {"name":"alpha","obj_fun":"ESJD","par_type":"slow"}

#medium_opt_metadata_argument = opt_default_arguments(name_list=["evolve_L","alpha"],par_type="medium",bounds_list=[(1,10),(0.1,1e6)])
#medium_opt_metadata_argument = opt_default_arguments(name_list=["evolve_L","alpha"],par_type="medium")

# gpyopt parameters input format
#gpyopt_slow_metadata_argument = {"obj_fun":"ESJD","par_type":"slow","name":"gpyopt","params":("evolve_L","alpha")}
#gpyopt_medium_metadata_argument = {"obj_fun":"ESJD","par_type":"medium","name":"gpyopt","params":("evolve_t")}
#gpyopt_fast_metatdata_argument = {"obj_fun":"ESJD","par_type":"fast","name":"gpyopt"}



#dual_arguments = [ep_dual_metadata_argument,evolve_L_opt_metadata_argument,alpha_opt_metadata_argument]

dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()

tune_settings_dict = tuning_settings(dual_args_list,[],[],other_arguments)

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()


sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

out = sampler1.start_sampling()