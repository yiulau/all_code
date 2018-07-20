from final_experiments.effect_of_scaling.setup import setup_scale_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}

save_name_unscaled = "unscaled_results.npz"
save_name_scaled = "scaled_results.npz"

num_units_list = [20,50,90]

setup_scale_experiment(num_unit_list=num_units_list,scaled=False,train_set=train_set,test_set=test_set,save_name=save_name_unscaled,seed=seed)

setup_scale_experiment(num_unit_list=num_units_list,scaled=True,train_set=train_set,test_set=test_set,save_name=save_name_scaled,seed=seed)
