# optimize network many times to obtain optima
# combine them and carry out prediction as if they came from a chain
# compare predictive accuracy with chain obtained from mcmc
from distributions.neural_nets.fc_V_model_4 import V_fc_model_4
from distributions.neural_nets.fc_V_model_4 import V_fc_model_4
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error
import numpy,torch,time
from final_experiments.ensemble_vs_mcmc.util import gradient_descent

def setup_ensemble_experiment(num_unit_list,list_num_ensemble_pts,train_set,validate_set,test_set,save_name,seed=1):


    output_names = ["ensemble_train_error","ensemble_te","ensemble_train_error_sd","ensemble_te_sd"]
    output_store = numpy.zeros((len(num_unit_list),len(list_num_ensemble_pts),len(output_names)))
    diagnostics_store = numpy.zeros(shape=[len(num_unit_list)]+[4,13])

    numpy.random.seed(seed)
    torch.manual_seed(seed)
    diver_store = numpy.zeros((len(num_unit_list),len(list_num_ensemble_pts)))
    time_store = numpy.zeros((len(num_unit_list),len(list_num_ensemble_pts)))

    for i in range(len(num_unit_list)):

        model_dict = {"num_units": num_unit_list[i]}
        for k in range(len(list_num_ensemble_pts)):
            start_time = time.time()
            prior_dict = {"name": "normal"}

            num_ensemble_pts = list_num_ensemble_pts[k]
            num_diver = 0
            ensemble_list = []
            v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_model_4, prior_dict=prior_dict,
                                                       input_data=train_set, model_dict=model_dict)


            for j in range(num_ensemble_pts):


                out, explode_grad = gradient_descent(number_of_iter=2000, lr=0.001,validation_set=validate_set,
                                                     v_obj=v_generator(precision_type="torch.DoubleTensor"))
                if explode_grad:
                    num_diver+= 1
                else:
                    ensemble_list.append(out.point_clone())



            ensemble_pts = numpy.zeros((len(ensemble_list),len(ensemble_list[0].flattened_tensor)))
            for z in range(len(ensemble_list)):
                ensemble_pts[z,:] = ensemble_list[z].flattened_tensor.numpy()

            ensemble_te, predicted, ensemble_te_sd = test_error(test_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                              mcmc_samples=ensemble_pts, type="classification",
                                              memory_efficient=False)
            ensemble_train_error, _, ensemble_train_error_sd = test_error(train_set, v_obj=v_generator(precision_type="torch.DoubleTensor"),
                                                        mcmc_samples=ensemble_pts, type="classification",
                                                        memory_efficient=False)

            total_time = time.time() - start_time


            output_store[i,k, 0] = ensemble_train_error
            output_store[i,k, 1] = ensemble_te
            output_store[i,k, 2] = ensemble_train_error_sd
            output_store[i,k, 3] = ensemble_te_sd
            diver_store[i,k] = num_diver
            time_store[i,k] = total_time

            #print(output_store)
    to_store = { "output": output_store,
    "output_names": output_names, "num_unit_list": num_unit_list, "seed": seed,"list_num_ensemble_pts":list_num_ensemble_pts,"num_div":diver_store,"time_store":time_store}
    numpy.savez(save_name, **to_store)
    return()