import pickle,os

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from experiments.experiment_obj import tuneinput_class

from experiments.correctdist_experiments.prototype import check_mean_var

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=3100,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=2000,
                                   warmup_per_chain=2100,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":["opt"],"second_order":[False],
              "evolve_L":["opt"],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

# ep_dual_metadata_argument = {"name":"epsilon","target":0.65,"gamma":0.05,"t_0":10,
#                         "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}

#ep_dual_metadata_argument = dual_default_arguments(name="epsilon")

#evolve_L_opt_metadata_argument = {"name":"evolve_L","obj_fun":"ESJD","bounds":(1,10),"par_type":"medium"}

#alpha_opt_metadata_argument = {"name":"alpha","obj_fun":"ESJD","par_type":"slow"}

medium_opt_metadata_argument = opt_default_arguments(name_list=["evolve_L","epsilon"],par_type="medium",obj_fun="ESJD_g_normalized",bounds_list=[(1,40),(0.001,0.2)])
#medium_opt_metadata_argument = opt_default_arguments(name_list=["evolve_L","alpha"],par_type="medium")

print(medium_opt_metadata_argument)

# gpyopt parameters input format
#gpyopt_slow_metadata_argument = {"obj_fun":"ESJD","par_type":"slow","name":"gpyopt","params":("evolve_L","alpha")}
#gpyopt_medium_metadata_argument = {"obj_fun":"ESJD","par_type":"medium","name":"gpyopt","params":("evolve_t")}
#gpyopt_fast_metatdata_argument = {"obj_fun":"ESJD","par_type":"fast","name":"gpyopt"}



#dual_arguments = [ep_dual_metadata_argument,evolve_L_opt_metadata_argument,alpha_opt_metadata_argument]

#dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()
opt_arguments = [medium_opt_metadata_argument]
tune_settings_dict = tuning_settings([],opt_arguments,[],other_arguments)

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()


sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

out = sampler1.start_sampling()