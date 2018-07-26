from final_experiments.ensemble_vs_mcmc.setup import setup_ensemble_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}
validate_set = {"input":input_data["input"][500:1000,:],"target":input_data["target"][500:1000]}
save_name = "ensemble_results.npz"

num_units_list = [35]
list_num_ensemble_pts = [2,10,50,100,1000]

setup_ensemble_experiment(num_unit_list=num_units_list,list_num_ensemble_pts=list_num_ensemble_pts,train_set=train_set,validate_set=validate_set,test_set=test_set,save_name=save_name,seed=seed)

