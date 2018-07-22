from final_experiments.adapt_cov.setup import setup_adapt_cov_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "adapt_cov_results.npz"


priors_list = ["normal","gaussian_inv_gamma_2","gaussian_inv_gamma_ard"]

setup_adapt_cov_experiment(priors_list=priors_list,train_set=train_set,test_set=test_set,save_name=save_name,seed=seed)
