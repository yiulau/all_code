import numpy
from experiments.tuning_method_experiments.opt_find_init_ep_L import opt_state
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.experiment_util import get_ess_and_esjds

def opt_experiment_ep_t(v_fun_list,ep_list,evolve_L_list,num_of_opt_steps,objective,input_dict,max_L):
    # given list of v_fun, epsilon, evolve_t's, number of bayesian optimization steps, an objective function
    # and an input dict , find optimal (ep,t) by repeatedly trying different combinations , sample long chain and
    # compare performance using different objective functions
    assert objective in ("median_ess_normalized","max_ess_normalized","min_ess_normalized","median_ess","max_ess",
                         "min_ess","esjd","esjd_normalized")


    num_grid_divides = len(ep_list)

    ep_bounds = [ep_list[0],ep_list[-1]]
    evolve_L_bounds = [evolve_L_list[0],evolve_L_list[-1]]

    chosen_init = [ep_list[numpy.asscalar(numpy.random.choice(num_grid_divides, 1))],
                   evolve_L_list[numpy.asscalar(numpy.random.choice(num_grid_divides, 1))]]

    this_opt_state = opt_state(bounds=[ep_bounds, evolve_L_bounds], init=chosen_init)
    cur_ep = chosen_init[0]
    cur_evolve_L = chosen_init[1]
    input_dict.update({"epsilon":[cur_ep],"evolve_L":[cur_evolve_L]})
    for j in range(num_of_opt_steps):
        mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=300, num_chains=4, num_cpu=4, thin=1,
                                               tune_l_per_chain=0,warmup_per_chain=150,is_float=False,
                                               isstore_to_disk=False,allow_restart=True,max_num_restarts=5)
        tune_settings_dict = tuning_settings([], [], [], [])
        input_dict.update({"epsilon":[cur_ep],"evolve_L":[cur_evolve_L]})
        tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

        sampler = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)
        sampler.start_sampling()
        out = get_ess_and_esjds(ran_sampler=sampler)

        #L = max(1,round(cur_evolve_t/cur_ep))
        # need to use actual number of transitions
        this_opt_state.update(new_y=-out[objective])
        cur_ep = this_opt_state.X_step[-1][0]
        cur_evolve_L = round(this_opt_state.X_step[-1][1])


    return(this_opt_state)