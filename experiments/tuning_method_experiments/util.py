import numpy
from experiments.tuning_method_experiments.opt_find_init_ep_L import opt_state
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.experiment_util import get_ess_and_esjds

# used in opt_find_init_ep_L
def opt_experiment_ep_t(v_fun_list,ep_list,evolve_t_list,num_of_opt_steps,objective,input_dict):
    # given list of v_fun, epsilon, evolve_t's, number of bayesian optimization steps, an objective function
    # and an input dict , find optimal (ep,t) by repeatedly trying different combinations , sample long chain and
    # compare performance using different objective functions
    assert objective in ("min_ess","esjd_normalized")

    num_grid_divides = len(ep_list)

    ep_bounds = [ep_list[0],ep_list[-1]]
    evolve_t_bounds = [evolve_t_list[0],evolve_t_list[-1]]
    chosen_init = [ep_list[numpy.asscalar(numpy.random.choice(num_grid_divides, 1))],
                   evolve_t_list[numpy.asscalar(numpy.random.choice(num_grid_divides, 1))]]

    this_opt_state = opt_state(bounds=[ep_bounds, evolve_t_bounds], init=chosen_init)
    cur_ep = chosen_init[0]
    cur_evolve_t = chosen_init[1]
    for j in range(num_of_opt_steps):
        mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=10000, num_chains=4, num_cpu=1, thin=1,
                                               tune_l_per_chain=0,warmup_per_chain=1000,is_float=False,
                                               isstore_to_disk=False,allow_restart=True,max_num_restarts=5)
        tune_settings_dict = tuning_settings([], [], [], [])
        input_dict.update({"epsilon":cur_ep,"evolve_t":cur_evolve_t})
        tune_dict = tuneinput_class(input_dict).singleton_tune_dict()
        sampler = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)
        sampler.start_sampling()
        out = get_ess_and_esjds(ran_sampler=sampler)
        this_opt_state.update(new_y=-out[objective])
        cur_ep = this_opt_state.X_step[-1][0]
        cur_evolve_t = this_opt_state.X_step[-1][1]


    return(this_opt_state)

def convert_to_numpy_results(out):
    # for exepriment grid vs opt
    return()



def min_ess_gnuts(v_fun,ep):
    mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=10000, num_chains=4, num_cpu=1, thin=1,
                                           tune_l_per_chain=0, warmup_per_chain=1000, is_float=False,
                                           isstore_to_disk=False, allow_restart=True, max_num_restarts=5)
    tune_settings_dict = tuning_settings([], [], [], [])
    input_dict = {"v_fun": [v_fun], "epsilon": [ep], "second_order": [False],
                    "metric_name": ["unit_e"], "dynamic": [True], "windowed": [None],
                  "criterion": ["gnuts"]}

    tune_dict = tuneinput_class(input_dict).singleton_tune_dict()
    sampler = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta, tune_settings_dict=tune_settings_dict)
    sampler.start_sampling()
    out = get_min_ess_and_esjds(ran_sampler=sampler)
    return(out)
def choose_optimal_L(v_fun,fixed_ep,L_list,windowed):

    store_min_ess = [None]*len(L_list)
    for i in range(len(L_list)):
        L = L_list[i]
        mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=10000, num_chains=4, num_cpu=1, thin=1,
                                               tune_l_per_chain=0, warmup_per_chain=1000, is_float=False,
                                               isstore_to_disk=False, allow_restart=True, max_num_restarts=5)
        tune_settings_dict = tuning_settings([], [], [], [])
        input_dict = {"v_fun": [v_fun], "epsilon": [fixed_ep], "second_order": [False],
                      "evolve_L": [L], "metric_name": ["unit_e"], "dynamic": [False], "windowed": [windowed],
                      "criterion": [None]}

        tune_dict = tuneinput_class(input_dict).singleton_tune_dict()
        sampler = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta, tune_settings_dict=tune_settings_dict)
        sampler.start_sampling()
        out = get_min_ess_and_esjds(ran_sampler=sampler)

        store_min_ess[i] = out["min_ess"]

    best_min_ess = max(store_min_ess)
    best_L = L_list[numpy.argmax(store_min_ess)]

    out = {"best_min_ess":best_min_ess,"best_L":best_L}
    return(out)



