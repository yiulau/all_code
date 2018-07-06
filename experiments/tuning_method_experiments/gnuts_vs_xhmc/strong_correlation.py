# each model has
from experiments.experiment_util import wishart_for_cov
from distributions.mvn import V_mvn
from experiments.experiment_obj import experiment_setting_dict,experiment
from experiments.experiment_obj import tuneinput_class
import numpy
# correlated normal
# use wisahrt for precision matrix for multivariate normal
# 1-pl item response theory model

# logistic regression

# neural network model - normal prior for all weights, scaled by number incoming units
# 8x8 logit

# rhorseshoe toy problem

# rhorseshoe logit problem
#
# compare median ESS/num_leapfrogs, median ESS
# same dual averaging for epsilon , acceptance rate set at 0.8, adaptively update diagonal cov matrix
# diagnostics (min,max,median ess, acceptance rate per chain, energy diagnostics)
#wishart_for_cov(dim=50)

# set up v_fun for each problem
num_repeats = 50
v_fun_list = []
####################################################################################################################################
num_chains_per_sampler = 4
np_store_gnuts = numpy.zeros(num_repeats,num_chains_per_sampler,30)
for i in range(num_repeats):
    experiment_setting_gnuts = experiment_setting_dict(chain_length=10000,num_chains_per_sampler=num_chains_per_sampler,warm_up=2000,
                                                 tune_l=1000,allow_restart=True,max_num_restarts=5,num_cpu_per_sampler=4)

    input_dict_gnuts = {"v_fun":v_fun_list,"epsilon":["dual"],"second_order":[False],"Cov":["adapt"],
                  "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}

    input_object_gnuts = tuneinput_class(input_dict_gnuts)
    experiment_instance_gnuts = experiment(input_object=input_object_gnuts,experiment_setting=experiment_setting_gnuts,fun_per_sampler=function)

    experiment_instance_gnuts.run()

    np_store_gnuts[i,:,:] = experiment_instance_gnuts.np_output()

#######################################################################################################################################
np_store_xhmc = numpy.zeros(num_repeats,num_chains_per_sampler,30)
for _ in range(num_repeats):
    experiment_setting_xhmc = experiment_setting_dict(chain_length=10000,num_chains_per_sampler=num_chains_per_sampler,warm_up=2000,
                                                 tune_l=1000,allow_restart=True,max_num_restarts=5,num_cpu_per_sampler=4)

    input_dict_xhmc = {"v_fun":v_fun_list,"epsilon":["dual"],"second_order":[False],"Cov":["adapt"],"xhmc_delta":[0.01,0.05,0.1],
                  "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["xhmc"]}

    input_object_xhmc = tuneinput_class(input_dict_xhmc)
    experiment_instance_xhmc = experiment(input_object=input_object_xhmc,experiment_setting=experiment_setting_xhmc,fun_per_sampler=function)

    experiment_instance_xhmc.run()
    np_store_xhmc[_, :, :] = experiment_instance_xhmc.np_output()
###########################################################################################################################################


numpy.savez([np_store_gnuts,np_store_xhmc])