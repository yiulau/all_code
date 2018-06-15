from experiments.experiment_obj import experiment_setting_dict,experiment
from experiments.experiment_obj import tuneinput_class
import numpy
v_fun_list = []

num_grid_divides = 20


ep_bounds = (1e-2,0.1)

L_bounds = (5,1000)

converted_t_bounds = (min(L_bounds)*min(ep_bounds),max(L_bounds)*max(ep_bounds))

ep_list = list(numpy.linspace(ep_bounds[0],ep_bounds[1],num_grid_divides))
evolve_L_list = list(numpy.linspace(L_bounds[0],L_bounds[1],num_grid_divides))
evolve_t_list = list(numpy.linspace(converted_t_bounds[0],converted_t_bounds[1],num_grid_divides))

#print(converted_t_bounds)

#####################################################################################################################################
experiment_setting_ep_L = experiment_setting_dict(chain_length=10000,num_chains_per_sampler=4,warm_up=1000,
                                             tune_l=0,allow_restart=True,max_num_restarts=5)

input_dict_ep_L = {"v_fun":v_fun_list,"epsilon":ep_list,"second_order":[False],
              "evolve_t":evolve_t_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_object_ep_L = tuneinput_class(input_dict_ep_L)
experiment_instance_ep_L = experiment(input_object=input_object_ep_L,experiment_setting=experiment_setting_ep_L,fun_per_sampler=function)

experiment_instance_ep_L.run()

result_grid_ep_L= experiment_instance_ep_L.experiment_result_grid_obj

##########################################################################################################################################
experiment_setting_ep_t = experiment_setting_dict(chain_length=10000,num_chains_per_sampler=4,warm_up=1000,
                                             tune_l=0,allow_restart=True,max_num_restarts=5)

input_dict_ep_t = {"v_fun":v_fun_list,"epsilon":ep_list,"second_order":[False],
              "evolve_t":evolve_t_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_object_ep_t = tuneinput_class(input_dict_ep_t)
experiment_instance_ep_t = experiment(input_object=input_object_ep_t,experiment_setting=experiment_setting_ep_t,fun_per_sampler=function)

experiment_instance_ep_t.run()

result_grid_ep_t= experiment_instance_ep_t.experiment_result_grid_obj






