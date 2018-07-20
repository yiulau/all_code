from distributions.neural_nets.fc_V_model_4 import V_fc_model_4
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error
import numpy,torch
from abstract.abstract_class_point import point
from experiments.float_vs_double.stability.leapfrog_stability import generate_q_list, generate_Hams, \
        leapfrog_stability_test
import dill as pickle

def setup_float_v_double_experiment(priors_list,train_set,test_set,save_name,seed=1):
    output_names = ["train_error", "test_error","train_error_sd","test_error_sd","min_ess","median_ess"]
    output_store = numpy.zeros((len(priors_list), 2,len(output_names)))

    diagnostics_store = numpy.zeros(shape=[len(priors_list),2]+[4,13])

    for i in range(len(priors_list)):
        for j in range(2):
            v_fun = V_fc_model_4

            prior_dict = {"name":priors_list[i]}
            model_dict = {"num_units":15}
            v_generator = wrap_V_class_with_input_data(class_constructor=v_fun, input_data=train_set,prior_dict=prior_dict,
                                                   model_dict=model_dict)

            if j==0:
                mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=1500, num_chains=4, num_cpu=4, thin=1,
                                                   tune_l_per_chain=800,
                                                   warmup_per_chain=900, is_float=False, isstore_to_disk=False,
                                                   allow_restart=True,seed=seed+i+1)
            else:
                mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=1500, num_chains=4, num_cpu=4,
                                                      thin=1,
                                                      tune_l_per_chain=800,
                                                      warmup_per_chain=900, is_float=True, isstore_to_disk=False,
                                                      allow_restart=True, seed=seed + i + 2)

            input_dict = {"v_fun": [v_generator], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
                          "max_tree_depth": [6],
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
            np_diagnostics,feature_names = sampler1.np_diagnostics()

            mcmc_samples_mixed = sampler1.get_samples(permuted=True)
            te, predicted,te_sd = test_error(test_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                         mcmc_samples=mcmc_samples_mixed, type="classification", memory_efficient=False)
            train_error,_,train_error_sd = test_error(train_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                         mcmc_samples=mcmc_samples_mixed, type="classification", memory_efficient=False)

            output_store[i,j,0] = train_error
            output_store[i,j,1] = te
            output_store[i,j,2] = train_error_sd
            output_store[i,j,3] = te_sd

            diagnostics_store[i,j,:,:] = np_diagnostics
            output_store[i,j,4] = np_diagnostics[0,10]
            output_store[i,j,5] = np_diagnostics[0,11]



    to_store = {"diagnostics":diagnostics_store,"output":output_store,"diagnostics_names":feature_names,"output_names":output_names}

    numpy.savez(save_name,**to_store)


    return()


def stability_experiment(priors_list,input_data,num_of_pts,save_name):

    store_outcome = numpy.zeros(shape=[len(priors_list),2,num_of_pts])

    stored = True
    for i in range(len(priors_list)):
        model_dict = {"num_units":20}
        prior_dict = {"name":priors_list[i]}
        v_fun = wrap_V_class_with_input_data(class_constructor=V_fc_model_4,prior_dict=prior_dict,model_dict=model_dict,
                                             input_data=input_data)

        out = generate_q_list(v_fun=v_fun, num_of_pts=num_of_pts)
        list_q_double = out["list_q_double"]
        list_q_float = out["list_q_float"]
        #print(list_q_double[0].flattened_tensor)
        #print(list_q_float[0].flattened_tensor)


        list_p_double = [None] * len(list_q_double)
        list_p_float = [None] * len(list_q_float)

        for j in range(len(list_q_float)):
            print(list_q_double[j].flattened_tensor)
            #print(list_q_double[j].list_tensor)
            p_double = list_q_double[j].point_clone()
            #print(p_double.flattened_tensor)

            momentum = torch.randn(len(p_double.flattened_tensor)).type("torch.DoubleTensor")
            #print(momentum)
            #print(p_double.flattened_tensor)
            #exit()
            p_double.flattened_tensor.copy_(momentum)
            p_double.load_flatten()
            p_float = list_q_float[j].point_clone()
            p_float.flattened_tensor.copy_(momentum.type("torch.FloatTensor"))
            p_float.load_flatten()

            list_p_double[j] = p_double
            list_p_float[j] = p_float


        out = generate_Hams(v_fun=v_fun)

        Ham_float = out["float"]
        Ham_double = out["double"]

        #print(list_q_double[0].flattened_tensor)
        #print(list_p_double[0].flattened_tensor)

        out_double = leapfrog_stability_test(Ham=Ham_double, epsilon=0.001, L=500, list_q=list_q_double,
                                             list_p=list_p_double, precision_type="torch.DoubleTensor")
        #print(out_double)

        out_float = leapfrog_stability_test(Ham=Ham_float, epsilon=0.001, L=500, list_q=list_q_float, list_p=list_p_float,
                                            precision_type="torch.FloatTensor")
        #print(out_float)

        #print(out_double.shape)
        #print(store_outcome.shape)
        store_outcome[i,0,:] = out_double
        store_outcome[i,1,:] = out_float

    to_store = { "output": store_outcome}
    numpy.savez(save_name,**to_store)
    return()