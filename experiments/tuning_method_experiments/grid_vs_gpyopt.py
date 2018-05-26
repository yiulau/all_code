
# compare best found by grid search with best found by gpyopt
# both using similar computational resources
# do the grid search first, then compute total computational resources used.
# start bayesian optimization looking at one point at a time. accumulate total computational resources. continue until
# exceeded previous total computational resources
#
# repeat experiment 10-20 times on same model,same data,same integrator,
# plot H,V,T to see if optimal t is roughly half-period , identify optimal t on graph for easy inspection
# see if optimal from gpyopt is systematically better than grid search
# find first point from gpyopt that beats grid search. and identify computational resources used up to that point
# cost of bayesian optimization is insignificant cuz dimension is 2 , maybe 3 and each point is equivalent to many samples <=>
# leapfrogs steps
# sample models:
# logistic (different data)
# hierarchical logistic
# funnel
# 8 schools (ncp)
# multivariate normal
# banana
# integrator
# unit_e hmc (ep,t) windowed_option
# dense_e , diag_e hmc (ep,t) adapting cov, or cov_diag at the same time windowed option
# softabs - diag,outer_product,diag_outer_product static (ep,t,alpha)
# xhmc - delta , unit_e, dense_e, diag_e, softabs (ep,delta) or (ep,delta,alpha)
#
# performance assessement
# objective function esjd/cost= number of gradients or esjd/seconds
#  ess (min, max , median)


# test for sensitivities to objective functions
# change objective functions keep everything the same

#model vs integrator

# test for

# experiment 3 variables (model,integrator,objective function)
# 3 x 3 numpy matrix. storing experiment information. depending on volume store the chain or just the experiment output



# supertransitions . given supertranstions = 10000 leapfrogs. for each L convert to number of transitions

def supertransitions(super_trans,L):
    num_transitions = round(super_trans/L)
    return(num_transitions)

# so that each point in the grid uses the same number of leapfrog

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

import numpy
from distributions.funnel_cp import V_funnel_cp
from distributions.funnel_ncp import V_funnel_ncp

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class

from experiments.correctdist_experiments.prototype import check_mean_var

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=500,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=100,is_float=False,isstore_to_disk=False)


num_grid_divides = 20
ep_list = list(numpy.linspace(1e-2,0.1,num_grid_divides))
evolve_t_list = list(numpy.linspace(0.15,1.5,num_grid_divides))

input_dict = {"v_fun":[V_funnel_cp],"epsilon":ep_list,"second_order":[False],
              "evolve_t":evolve_t_list,"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict2 = {"v_fun":[V_funnel_ncp],"epsilon":["opt"],"second_order":[False],
              "evolve_t":["opt"],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

medium_opt_metadata_argument = opt_default_arguments(name_list=["evolve_t","epsilon"],par_type="medium",bounds_list=[(0.15,1.5),(0.01,0.1)])

opt_arguments = [medium_opt_metadata_argument]

other_arguments = other_default_arguments()

tuning_settings_dict = tuning_settings([],opt_arguments,[],other_arguments)

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

