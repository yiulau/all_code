from final_experiments.num_layers.setup import setup_num_layers_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 120
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "num_layers_results.npz"


num_layers_list = [2,3,4]

setup_num_layers_experiment(num_layers_list=num_layers_list,train_set=train_set,test_set=test_set,
                               save_name=save_name,seed=seed)

