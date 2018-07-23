from final_experiments.hmc_v_gnuts.setup import setup_hmc_windowed_experiment
from input_data.convert_data_to_dict import get_data_dict


seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "hmc_windowed_results.npz"

L_list = [10,50,100,200,500]
#L_list = [10]

setup_hmc_windowed_experiment(L_list=L_list,train_set=train_set,test_set=test_set,
                               save_name=save_name,seed=seed)

