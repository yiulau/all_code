from distributions.neural_nets.fc_V_model_4 import V_fc_model_4
from distributions.neural_nets.fc_V_model_1 import V_fc_model_1
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error
import numpy,time

def setup_scale_experiment(num_unit_list,scaled,train_set,test_set,save_name,seed=1):

    output_names = ["train_error", "test_error","train_error_sd","test_error_sd"]
    output_store = numpy.zeros((len(num_unit_list), len(output_names)))
    diagnostics_store = numpy.zeros(shape=[len(num_unit_list)]+[4,13])
    time_list = []
    for i in range(len(num_unit_list)):
        start_time = time.time()
        if scaled:
            v_fun = V_fc_model_1
        else:
            v_fun = V_fc_model_4

        prior_dict = {"name":"normal"}
        model_dict = {"num_units":num_unit_list[i]}
        v_generator = wrap_V_class_with_input_data(class_constructor=v_fun, input_data=train_set,prior_dict=prior_dict,
                                                   model_dict=model_dict)
        mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=2000, num_chains=4, num_cpu=4, thin=1,
                                               tune_l_per_chain=900,
                                               warmup_per_chain=1000, is_float=False, isstore_to_disk=False,
                                               allow_restart=True,seed=seed)

        input_dict = {"v_fun": [v_generator], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
                      "max_tree_depth": [8],
                      "metric_name": ["diag_e"], "dynamic": [True], "windowed": [False], "criterion": ["xhmc"],"xhmc_delta":[0.1]}

        ep_dual_metadata_argument = {"name": "epsilon", "target": 0.9, "gamma": 0.05, "t_0": 10,
                                     "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}
        # #
        adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=v_generator(
            precision_type="torch.DoubleTensor").get_model_dim())]
        dual_args_list = [ep_dual_metadata_argument]
        other_arguments = other_default_arguments()
        tune_settings_dict = tuning_settings(dual_args_list, [], adapt_cov_arguments, other_arguments)
        tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

        sampler1 = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta, tune_settings_dict=tune_settings_dict)

        sampler1.start_sampling()
        total_time = time.time() - start_time
        np_diagnostics,feature_names = sampler1.np_diagnostics()

        mcmc_samples_mixed = sampler1.get_samples(permuted=True)
        te, predicted,te_sd = test_error(test_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                     mcmc_samples=mcmc_samples_mixed, type="classification", memory_efficient=False)
        train_error,_,train_error_sd = test_error(train_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                     mcmc_samples=mcmc_samples_mixed, type="classification", memory_efficient=False)

        output_store[i,0] = train_error
        output_store[i,1] = te
        output_store[i,2] = train_error_sd
        output_store[i,3] = te_sd
        diagnostics_store[i,:,:] = np_diagnostics

        time_list.append(total_time)





    to_store = {"diagnostics":diagnostics_store,"output":output_store,"diagnostics_names":feature_names,"output_names":output_names,"num_unit_list":num_unit_list,"seed":seed}

    numpy.savez(save_name,**to_store)


    return()
