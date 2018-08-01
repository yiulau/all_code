from distributions.neural_nets.fc_V_model_4 import V_fc_model_4
from post_processing.ESS_nuts import diagnostics_stan
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error
import numpy,torch,time
from abstract.abstract_class_point import point
from experiments.float_vs_double.stability.leapfrog_stability import generate_q_list, generate_Hams, \
        leapfrog_stability_test
from experiments.neural_net_experiments.gibbs_vs_joint_sampling.V_hierarchical_fc1 import V_fc_gibbs_model_1
from general_util.pytorch_random import generate_gamma
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_class_point import point
from abstract.mcmc_sampler import log_class
from experiments.neural_net_experiments.gibbs_vs_joint_sampling.gibbs_vs_together_hyperparam import update_param_and_hyperparam_one_step,update_param_and_hyperparam_dynamic_one_step

def setup_gibbs_v_joint_experiment(num_units_list,train_set,test_set,num_samples,save_name,seed=1):
    output_names = ["train_error", "test_error","train_error_sd","test_error_sd","sigma_2_ess","mean_sigma2","median_sigma2","min_ess","median_ess"]
    output_store = numpy.zeros((len(num_units_list), 3,len(output_names)))

    diagnostics_store = numpy.zeros(shape=[len(num_units_list),3]+[4,13])
    time_store = numpy.zeros(shape=[len(num_units_list),3])
    for i in range(len(num_units_list)):
        for j in range(3):
            start_time = time.time()
            v_fun = V_fc_model_4
            model_dict = {"num_units":num_units_list[i]}

            mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=1000+num_samples, num_chains=4, num_cpu=4, thin=1,
                                                   tune_l_per_chain=900,
                                                   warmup_per_chain=1000, is_float=False, isstore_to_disk=False,
                                                   allow_restart=True, seed=seed + i + 1)
            if j==2:
                v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_gibbs_model_1, input_data=train_set,model_dict=model_dict)
                v_obj = v_generator(precision_type="torch.DoubleTensor", gibbs=True)
                metric_obj = metric(name="unit_e", V_instance=v_obj)
                Ham = Hamiltonian(v_obj, metric_obj)

                init_q_point = point(V=v_obj)
                init_hyperparam = torch.abs(torch.randn(1)) + 3
                log_obj = log_class()


                dim = len(init_q_point.flattened_tensor)
                mcmc_samples_weight = torch.zeros(1, num_samples+1000, dim)
                mcmc_samples_hyper = torch.zeros(1, num_samples+1000, 1)
                for iter in range(num_samples+1000):
                    print("iter {}".format(iter))
                    outq, out_hyperparam = update_param_and_hyperparam_dynamic_one_step(init_q_point, init_hyperparam,
                                                                                        Ham, 0.01, log_obj)
                    init_q_point.flattened_tensor.copy_(outq.flattened_tensor)
                    init_q_point.load_flatten()
                    init_hyperparam = out_hyperparam
                    mcmc_samples_weight[0, iter, :] = outq.flattened_tensor.clone()
                    mcmc_samples_hyper[0, iter, 0] = out_hyperparam

                mcmc_samples_weight = mcmc_samples_weight[:,1000:,:].numpy()
                mcmc_samples_hyper = mcmc_samples_hyper[:,1000:,:].numpy()

                te, predicted, te_sd = test_error(test_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                                  mcmc_samples=mcmc_samples_weight[0,:,:], type="classification",
                                                  memory_efficient=False)
                train_error, _, train_error_sd = test_error(train_set,
                                                            v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                                            mcmc_samples=mcmc_samples_weight[0,:,:], type="classification",
                                                            memory_efficient=False)
                sigma2_diagnostics = diagnostics_stan(mcmc_samples_hyper)
                sigma2_ess = sigma2_diagnostics["ess"]
                posterior_mean_hidden_in_sigma2 = numpy.mean(mcmc_samples_hyper)
                posterior_median_hidden_in_sigma2 = numpy.median(mcmc_samples_hyper)
                weight_ess = diagnostics_stan(mcmc_samples_weight)["ess"]

                min_ess = min(sigma2_ess,min(weight_ess))
                median_ess = numpy.median([sigma2_ess]+list(weight_ess))

                output_store[i, j, 0] = train_error
                output_store[i, j, 1] = te
                output_store[i, j, 2] = train_error
                output_store[i, j, 3] = te_sd
                output_store[i, j, 4] = sigma2_ess
                output_store[i, j, 5] = posterior_mean_hidden_in_sigma2
                output_store[i, j, 6] = posterior_median_hidden_in_sigma2
                output_store[i, j, 7] = min_ess
                output_store[i, j, 8] = median_ess

            elif j==0:
                prior_dict = {"name":"gaussian_inv_gamma_1"}
                v_generator = wrap_V_class_with_input_data(class_constructor=v_fun,input_data=train_set,prior_dict=prior_dict,model_dict=model_dict)

            elif j==1:
                prior_dict = {"name": "gaussian_inv_gamma_2"}
                v_generator = wrap_V_class_with_input_data(class_constructor=v_fun,input_data=train_set,prior_dict=prior_dict,model_dict=model_dict)




            if j == 0 or j==1:
                input_dict = {"v_fun": [v_generator], "epsilon": ["dual"], "second_order": [False],
                              "max_tree_depth": [8],
                              "metric_name": ["unit_e"], "dynamic": [True], "windowed": [False], "criterion": ["xhmc"],
                              "xhmc_delta": [0.1]}
                ep_dual_metadata_argument = {"name": "epsilon", "target": 0.9, "gamma": 0.05, "t_0": 10,
                                             "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}

                dual_args_list = [ep_dual_metadata_argument]
                other_arguments = other_default_arguments()
                tune_settings_dict = tuning_settings(dual_args_list, [], [], other_arguments)
                tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

                sampler1 = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta, tune_settings_dict=tune_settings_dict)

                sampler1.start_sampling()

                np_diagnostics,feature_names = sampler1.np_diagnostics()

                mcmc_samples_hidden_in = sampler1.get_samples_alt(prior_obj_name="hidden_in", permuted=False)
                samples = mcmc_samples_hidden_in["samples"]
                hidden_in_sigma2_indices = mcmc_samples_hidden_in["indices_dict"]["sigma2"]
                sigma2_diagnostics = diagnostics_stan(samples[:, :, hidden_in_sigma2_indices])
                sigma2_ess = sigma2_diagnostics["ess"]

                posterior_mean_hidden_in_sigma2 = numpy.mean(
                    samples[:, :, hidden_in_sigma2_indices].reshape(-1, len(hidden_in_sigma2_indices)), axis=0)
                posterior_median_hidden_in_sigma2 = numpy.median(
                    samples[:, :, hidden_in_sigma2_indices].reshape(-1, len(hidden_in_sigma2_indices)), axis=0)

                mcmc_samples_mixed = sampler1.get_samples(permuted=True)
                te, predicted,te_sd = test_error(test_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                             mcmc_samples=mcmc_samples_mixed, type="classification", memory_efficient=False)
                train_error,_,train_error_sd = test_error(train_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                             mcmc_samples=mcmc_samples_mixed, type="classification", memory_efficient=False)

                output_store[i,j,0] = train_error
                output_store[i,j,1] = te
                output_store[i,j,2] = train_error
                output_store[i,j,3] = te_sd
                output_store[i,j,4] = sigma2_ess
                output_store[i,j,5] = posterior_mean_hidden_in_sigma2
                output_store[i,j,6] = posterior_median_hidden_in_sigma2

                diagnostics_store[i,j,:,:] = np_diagnostics
                output_store[i,j,7] = np_diagnostics[0,10]
                output_store[i,j,8] = np_diagnostics[0,11]

            total_time = time.time() - start_time()
            time_store[i,j] = total_time



    to_store = {"diagnostics":diagnostics_store,"output":output_store,"diagnostics_names":feature_names,
                "output_names":output_names,"seed":seed,"num_units_list":num_units_list,"time_store":time_store}

    numpy.savez(save_name,**to_store)


    return()