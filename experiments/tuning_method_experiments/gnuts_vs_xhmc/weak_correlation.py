# iid normal

# gnuts vs xhmc

# dual averaging epsilon, diag cov metric

# compare median ESS/

# each model has
from experiments.experiment_util import wishart_for_cov
from distributions.mvn import V_mvn
from experiments.experiment_obj import experiment_setting_dict,experiment
from experiments.experiment_obj import tuneinput_class
from distributions.mvn import V_mvn
from experiments.tuning_method_experiments.gnuts_vs_xhmc.util import fun_extract_median_ess
import numpy
from abstract.util import wrap_V_class_with_input_data
#wishart_for_cov(dim=50)

# set up v_fun for each problem
num_repeats = 2

input_data = {"input":numpy.eye(5,5)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
V_mvn2 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
v_fun_list = [V_mvn1,V_mvn2]
target_fun = fun_extract_median_ess
####################################################################################################################################
num_chains_per_sampler = 4
np_store_gnuts = [None]*num_repeats
np_diagnostics_gnuts = [None]*num_repeats
for i in range(num_repeats):
    experiment_setting_gnuts = experiment_setting_dict(chain_length=5000,num_chains_per_sampler=4,warm_up=2000,
                                                 tune_l=1000,allow_restart=True,max_num_restarts=5,num_cpu_per_sampler=4)

    input_dict_gnuts = {"v_fun":v_fun_list,"epsilon":["dual"],"second_order":[False],"cov":["adapt"],
                  "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}

    input_object_gnuts = tuneinput_class(input_dict_gnuts)
    experiment_instance_gnuts = experiment(input_object=input_object_gnuts,experiment_setting=experiment_setting_gnuts,fun_per_sampler=target_fun)

    experiment_instance_gnuts.run()

    np_store,col_names,output_names = experiment_instance_gnuts.np_output()
    np_store_diagnostics,diagnostics_names = experiment_instance_gnuts.np_diagnostics()
    np_diagnostics_gnuts[i] = np_store_diagnostics
    np_store_gnuts[i] = np_store

np_store_gnuts = numpy.stack(np_store_gnuts,axis=0)
np_diagnostics_gnuts = numpy.stack(np_diagnostics_gnuts,axis=0)
exit()
#######################################################################################################################################
np_store_xhmc = [None]*num_repeats
np_diagnostics_xhmc = [None]*num_repeats
for _ in range(num_repeats):
    experiment_setting_xhmc = experiment_setting_dict(chain_length=10000,num_chains_per_sampler=4,warm_up=2000,
                                                 tune_l=1000,allow_restart=True,max_num_restarts=5,num_cpu_per_sampler=4)

    input_dict_xhmc = {"v_fun":v_fun_list,"epsilon":["dual"],"second_order":[False],"cov":["adapt"],"xhmc_delta":[0.01,0.05,0.1],
                  "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["xhmc"]}

    input_object_xhmc = tuneinput_class(input_dict_xhmc)
    experiment_instance_xhmc = experiment(input_object=input_object_xhmc,experiment_setting=experiment_setting_xhmc,fun_per_sampler=target_fun)

    experiment_instance_xhmc.run()
    np_store,col_names,output_names = experiment_instance_xhmc.np_output()
    np_store_diagnostics,diagnostics_names = experiment_instance_xhmc.np_diagnostics()
    np_diagnostics_xhmc[i] = np_store_diagnostics
    np_store_xhmc[i] = np_store

np_store_xhmc = numpy.stack(np_store_xhmc,axis=0)
np_diagnostics_xhmc = numpy.stack(np_diagnostics_xhmc,axis=0)
###########################################################################################################################################

result = {"gnuts":np_store_gnuts,"xhmc":np_store_xhmc,"gnuts_diagnostics":np_diagnostics_gnuts,"xhmc_diagnostics":np_diagnostics_xhmc}
result.update({"diagnostics_names":diagnostics_names})

