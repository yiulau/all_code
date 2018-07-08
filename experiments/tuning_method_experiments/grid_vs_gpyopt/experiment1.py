import numpy
from experiments.experiment_obj import experiment_setting_dict,experiment
from experiments.experiment_obj import tuneinput_class
from experiments.tuning_method_experiments.grid_vs_gpyopt.util import opt_experiment_ep_t
from experiments.tuning_method_experiments.util import convert_to_numpy_results

# experiment 1 (ep,L) Find best performance . identify L (calculate accumulate best using grid)
# compare with bayes opt when specifying constraint (upper bound on L)
# grid computed once. each with 4 chain
# opt computed 20 times. process and compare with grid output each time. Random initialization in hyperparameter space
# unit_e hmc


num_repeats = 2
num_grid_divides = 5

ep_bounds = [1e-2,0.1]
evolve_t_bounds = [0.15,50.]
# add constraints such that L = round(evolove_t/ep) < 1024
ep_list = list(numpy.linspace(ep_bounds[0],ep_bounds[1],num_grid_divides))
evolve_t_list = list(numpy.linspace(evolve_t_bounds[0],evolve_t_bounds[1],num_grid_divides))
v_fun_list = []

# grid computations
experiment_setting = experiment_setting_dict(chain_length=2000,num_chains_per_sampler=4,warm_up=1000,
                                             tune_l=0,allow_restart=True,max_num_restarts=5,num_cpu_per_sampler=4)

input_dict = {"v_fun":v_fun_list,"epsilon":ep_list,"second_order":[False],
              "evolve_t":evolve_t_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_object = tuneinput_class(input_dict)
experiment_instance = experiment(input_object=input_object,experiment_setting=experiment_setting,fun_per_sampler=function)

experiment_instance.run()
result_grid= experiment_instance.experiment_result_grid_obj
np_grid_store,col_names,output_names = experiment_instance.np_output()
grid_diagnostics,diagnostics_names = experiment_instance.np_diagnostics()
np_diagnostics_gnuts = grid_diagnostics

# find best ess/L . identify L
optimal_L = 1024


np_store_opt = numpy.zeros(num_repeats,2)
for i in range(num_repeats):
    opt_experiment_result_esjd_normalized = opt_experiment_ep_t(v_fun_list=v_fun_list,ep_list=ep_list,
                                                         evolve_t_list=evolve_t_list,
                                                         num_of_opt_steps=num_grid_divides*num_grid_divides,
                                                         objective="esjd_normalized",input_dict=input_dict)

    np_result = convert_to_numpy_results(opt_experiment_result_esjd_normalized)
    #result = {"esjd_normalized":opt_experiment_result_esjd_normalized}
    np_store_opt[i,:] = np_result


# after getting results compute accumulative num_leapfrog_steps
# get best performance attained first after exceeding total number of leapfrog in grid search
# get best performance overall
# get accumulative best performance
#out = {"grid_results":result_grid,"opt_results":result_opt_list}

#converted_to_np_results = convert_to_numpy_results(out)
save_address = "grid_experiment1_outcome.npz"
numpy.savez(save_address,allow_pickle=False)

