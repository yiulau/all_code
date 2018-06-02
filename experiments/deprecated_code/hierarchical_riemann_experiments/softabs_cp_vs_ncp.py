# compare softabs on ncp vs cp parametrization

# models:
#- funnel
# 8 schools
# horseshoe prior
# horseshoe prior plus
#
# fix (ep,t)
# integrators:
# gnuts option diag , windowed
# static option diag , windowed
#
# want to see if softabs works equally well in cp and ncp
# diagnostics
# ess


from distributions.funnel_cp import V_funnel_cp
from distributions.funnel_ncp import V_funnel_ncp

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.experiment_obj import experiment,experiment_setting_dict

from experiments.correctdist_experiments.prototype import check_mean_var

num_per_model = 20
mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=500,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=100,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_funnel_cp],"epsilon":[0.1],"alpha":[1e6,1e2],"second_order":[True],
              "evolve_L":[10],"metric_name":["softabs"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict2 = {"v_fun":[V_funnel_ncp],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_obj  = tuneinput_class(input_dict)

input_obj2 = tuneinput_class(input_dict2)

experiment_setting_dict = experiment_setting_dict(chain_length=10000,num_repeat=num_per_model)
experiment_obj = experiment(input_object=input_obj,experiment_setting=experiment_setting_dict)

experiment_obj.run()

experiment_obj2 = experiment(input_object=input_obj2,experiment_setting=experiment_setting_dict)

experiment_obj2.run()

