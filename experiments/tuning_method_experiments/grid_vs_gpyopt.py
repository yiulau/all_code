# update dynamically during tuning phase
# compare best found by grid search with best found by gpyopt
# both using similar computational resources
# do the grid search first, then compute total computational resources used.
# start bayesian optimization looking at one point at a time. accumulate total computational resources. continue until
# exceeded previous total computational resources
#
# repeat experiment 10-20 times on same model,same data,same integrator,
# see if optimal from gpyopt is systematically better than grid search
# find first point from gpyopt that beats grid search. and identify computational resources used up to that point
# cost of bayesian optimization is insignificant cuz dimension is 2 , maybe 3 and each point is equivalent to many samples <=>
# leapfrogs steps
# sample models:
# logistic (different data)
# multivariate normal
# neural network (normal prior scaled by number of incoming units)
# integrator = leapfrog
# unit_e hmc (ep,t)
# xhmc -  unit_e  (ep,delta)
#
# performance assessement
# objective function esjd/cost= number of gradients or esjd/seconds
#  ess (min, max , median)

#model vs integrator

# supertransitions . given supertranstions = 10000 leapfrogs. for each L convert to number of transitions

def supertransitions(super_trans,L):
    num_transitions = round(super_trans/L)
    return(num_transitions)

# so that each point in the grid uses the same number of leapfrog

# fix (ep,t)
# integrators:
# gnuts option diag , windowed
# static option diag , windowed
#
# diagnostics
# ess

import numpy,pickle,torch
from experiments.experiment_obj import experiment_setting_dict,experiment
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.tuning_method_experiments.util import opt_experiment_ep_t
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from experiments.tuning_method_experiments.util import convert_to_numpy_results
# mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=10000,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=0,
#                                    warmup_per_chain=1000,is_float=False,isstore_to_disk=False,allow_restart=True,max_num_restarts=5)
#

seed_id = 1
torch.manual_seed(seed_id)
numpy.random.seed(seed_id)


save_address = "grid_experiment_outcome.npz"
num_repeats = 50
num_grid_divides = 5

ep_bounds = [1e-2,0.1]
evolve_t_bounds = [0.15,5.]
ep_list = list(numpy.linspace(ep_bounds[0],ep_bounds[1],num_grid_divides))
evolve_t_list = list(numpy.linspace(evolve_t_bounds[0],evolve_t_bounds[1],num_grid_divides))
v_fun_list = []



#for i in range(num_repeats):

experiment_setting = experiment_setting_dict(chain_length=10000,num_chains_per_sampler=4,warm_up=1000,
                                             tune_l=0,allow_restart=True,max_num_restarts=5)

input_dict = {"v_fun":v_fun_list,"epsilon":ep_list,"second_order":[False],
              "evolve_t":evolve_t_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_object = tuneinput_class(input_dict)
experiment_instance = experiment(input_object=input_object,experiment_setting=experiment_setting,fun_per_sampler=function)

experiment_instance.run()

result_grid= experiment_instance.experiment_result_grid_obj



# with open(save_address, 'wb') as f:
#     pickle.dump(result_list, f)

# start opt part

result_opt_list = [None]*num_repeats
for i in range(num_repeats):
    opt_experiment_result_esjd_normalized = opt_experiment_ep_t(v_fun_list=v_fun_list,ep_list=ep_list,
                                                         evolve_t_list=evolve_t_list,
                                                         num_of_opt_steps=num_grid_divides*num_grid_divides,
                                                         objective="esjd_normalized",input_dict=input_dict)

    #result = {"esjd_normalized":opt_experiment_result_esjd_normalized}

    result_opt_list[i] = opt_experiment_result_esjd_normalized


out = {"grid_results":result_grid,"opt_results":result_opt_list}

converted_to_np_results = convert_to_numpy_results(out)

numpy.savez(save_address,allow_pickle=False)



#
# with open(save_address, 'wb') as f:
#     pickle.dump(out, f)


# for i in range(num_repeats):
#     chosen_init = [ep_list[numpy.asscalar(numpy.random.choice(num_grid_divides,1))],
#                    evolve_t_list[numpy.asscalar(numpy.random.choice(num_grid_divides,1))]]
#
#     this_opt_state = opt_state(bounds=[ep_bounds,evolve_t_bounds],init=chosen_init)
#     for j in range(num_grid_divides*num_grid_divides):





#






