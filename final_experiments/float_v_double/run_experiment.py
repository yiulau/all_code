from final_experiments.float_v_double.setup import setup_float_v_double_experiment,stability_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "float_vs_double_convergence_results.npz"


priors_list = ["normal","gaussian_inv_gamma_2"]

#setup_float_v_double_experiment(priors_list=priors_list,train_set=train_set,test_set=test_set,save_name=save_name,seed=seed)

save_name_stability = "float_vs_double_stability_results.npz"
stability_experiment(priors_list=priors_list,input_data=train_set,num_of_pts=50,save_name=save_name_stability)


