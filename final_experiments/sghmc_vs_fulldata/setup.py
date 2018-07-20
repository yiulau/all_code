from experiments.neural_net_experiments.sghmc_vs_batch_hmc.model import V_fc_model_1
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error
import numpy,torch
from post_processing.diagnostics import WAIC,convert_mcmc_tensor_to_list_points
from abstract.abstract_class_point import point
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error
from post_processing.ESS_nuts import diagnostics_stan
from final_experiments.sghmc_vs_fulldata.util import sghmc_sampler

def setup_sghmc_experiment(ep_list,L_list,eta_list,train_set,test_set,save_name,seed=1):
    output_names = ["train_error", "test_error","train_error_sd","test_error_sd"]
    output_store = numpy.zeros((len(ep_list),len(L_list),len(eta_list), len(output_names)))

    diagnostics_store = numpy.zeros(shape=[len(ep_list),len(L_list),len(eta_list)]+[4,13])
    model_dict = {"num_units":35}
    prior_dict = {"name":"normal"}

    for i in range(len(ep_list)):
        for j in range(len(L_list)):
            for k in range(len(eta_list)):
                v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_model_1, input_data=train_set,
                                                           prior_dict=prior_dict, model_dict=model_dict)

                v_obj = v_generator(precision_type="torch.DoubleTensor")
                metric_obj = metric(name="unit_e", V_instance=v_obj)
                Ham = Hamiltonian(V=v_obj, metric=metric_obj)

                full_data = train_set
                init_q_point = point(V=v_obj)
                out = sghmc_sampler(init_q_point=init_q_point, epsilon=ep_list[i], L=L_list[j], Ham=Ham, alpha=0.01, eta=eta_list[k],
                                    betahat=0, full_data=full_data, num_samples=2000, thin=0, burn_in=1000,
                                    batch_size=25)
                store = out[0]
                v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_model_1, input_data=train_set,
                                                           prior_dict=prior_dict, model_dict=model_dict)
                test_mcmc_samples = store.numpy()

                te1, predicted1,te_sd = test_error(test_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                             mcmc_samples=test_mcmc_samples, type="classification",
                                             memory_efficient=False)

                train_error, predicted1, train_error_sd = test_error(train_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                                    mcmc_samples=test_mcmc_samples, type="classification",
                                                    memory_efficient=False)


                output_store[i,j,k,0] = train_error
                output_store[i,j,k,1] = te1
                output_store[i,j,k,2] = train_error_sd
                output_store[i,j,k,3] = te_sd





    to_store = {"diagnostics":diagnostics_store,"output":output_store,"output_names":output_names,"seed":seed}

    numpy.savez(save_name,**to_store)


    return()