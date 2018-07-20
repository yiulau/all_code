from final_experiments.waic_test_error.setup import setup_waic_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "waic_results.npz"


num_units_list = [25,50,75,100]

setup_waic_experiment(num_units_list=num_units_list,train_set=train_set,test_set=test_set,
                               save_name=save_name,seed=seed)

