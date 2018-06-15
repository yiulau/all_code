from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class,experiment,experiment_setting_dict
import numpy


# mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=10000,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=0,
#                                    warmup_per_chain=1000,is_float=False,isstore_to_disk=False,allow_restart=True,max_num_restarts=5)


num_grid_divides = 2
ep_list = list(numpy.linspace(1e-2,0.1,num_grid_divides))
evolve_t_list = list(numpy.linspace(0.15,5.0,num_grid_divides))

v_fun_list = []
input_dict = {"v_fun":v_fun_list,"epsilon":ep_list,"second_order":[False],
              "evolve_t":evolve_t_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

experiment_setting = experiment_setting_dict(chain_length=10000,num_repeat=20,num_chains_per_sampler=4,warm_up=1000,
                                             tune_l=0,save_name="temp_experiment.pkl")


input_object = tuneinput_class(input_dict)
experiment_instance = experiment(input_object=input_object,experiment_setting=experiment_setting,fun_per_sampler=function)

experiment.run()


