import numpy
from experiments.experiment_obj import experiment_setting_dict,experiment
from experiments.tuning_method_experiments.grid_vs_gpyopt.util import opt_experiment_ep_t
from experiments.tuning_method_experiments.util import convert_to_numpy_results
from experiments.experiment_obj import tuneinput_class
from distributions.mvn import V_mvn
from experiments.tuning_method_experiments.gnuts_vs_xhmc.util import fun_extract_median_ess
from abstract.util import wrap_V_class_with_input_data
from experiments.experiment_util import wishart_for_cov
# experiment 1 (ep,L) Find best performance . identify L (calculate accumulate best using grid)
# compare with bayes opt when specifying constraint (upper bound on L)
# grid computed once. each with 4 chain
# opt computed 20 times. process and compare with grid output each time. Random initialization in hyperparameter space
# unit_e hmc
# make sure the chain is long enough to ensure consistent estimates of the ess say chain length = 20000
# plot ess/L standard deviation as well
# hmc need to add jitter (0.9 ep, 1.1 ep)

num_repeats = 2
num_grid_divides = 3

ep_bounds = [1e-3,0.2]
evolve_t_bounds = [0.15,50.]
evolve_L_bounds = [5,1000]
# add constraints such that L = round(evolove_t/ep) < 1024
ep_list = list(numpy.linspace(ep_bounds[0],ep_bounds[1],num_grid_divides))
evolve_t_list = list(numpy.linspace(evolve_t_bounds[0],evolve_t_bounds[1],num_grid_divides))
evolve_L_list = list(numpy.linspace(evolve_L_bounds[0],evolve_L_bounds[1],num_grid_divides))
input_data = {"input":wishart_for_cov(dim=10)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
V_mvn2 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
target_fun = fun_extract_median_ess
v_fun_list = [V_mvn1,V_mvn2]

# grid computations
# experiment_setting = experiment_setting_dict(chain_length=300,num_chains_per_sampler=4,warm_up=150,
#                                              tune_l=0,allow_restart=True,max_num_restarts=5,num_cpu_per_sampler=4)
#
input_dict = {"v_fun":v_fun_list[0:1],"epsilon":ep_list,"second_order":[False],"stepsize_jitter":[True],
              "evolve_L":evolve_L_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}
#
# input_object = tuneinput_class(input_dict)
# experiment_instance = experiment(input_object=input_object,experiment_setting=experiment_setting,fun_per_sampler=target_fun)
#
# experiment_instance.run()
# result_grid= experiment_instance.experiment_result_grid_obj
# np_grid_store,col_names,output_names = experiment_instance.np_output()
# grid_diagnostics,diagnostics_names = experiment_instance.np_diagnostics()
# np_diagnostics_gnuts = grid_diagnostics
#

# find best ess/total_num_transitions . identify L
optimal_L = 102


result_list = []
#np_store_opt = numpy.zeros(num_repeats,2)
for i in range(num_repeats):
    opt_experiment_result_median_ess = opt_experiment_ep_t(v_fun_list=v_fun_list,ep_list=ep_list,
                                                         evolve_L_list=evolve_L_list,
                                                         num_of_opt_steps=num_grid_divides*num_grid_divides,
                                                         objective="median_ess_normalized",input_dict=input_dict,max_L=optimal_L)

    result_list.append(opt_experiment_result_median_ess)
    #np_result = convert_to_numpy_results(opt_experiment_result_median_ess)
    #result = {"esjd_normalized":opt_experiment_result_esjd_normalized}
    #np_store_opt[i,:] = np_result


print(result_list[0].X_step)
print(result_list[0].Y_step)
print(result_list[1].X_step)
print(result_list[1].Y_step)
# after getting results compute accumulative num_leapfrog_steps
# get best performance attained first after exceeding total number of leapfrog in grid search
# get best performance overall
# get accumulative best performance
#out = {"grid_results":result_grid,"opt_results":result_opt_list}

#converted_to_np_results = convert_to_numpy_results(out)
save_address = "grid_experiment1_outcome.npz"
numpy.savez(save_address,allow_pickle=False)

