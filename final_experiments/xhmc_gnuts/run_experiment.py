from final_experiments.xhmc_gnuts.setup import setup_xhmc_gnuts_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "xhmc_gnuts_results.npz"

xhmc_delta_list = [0.01,0.05,0.1,0.2]
setup_xhmc_gnuts_experiment(xhmc_delta_list=[0.05],train_set=train_set,test_set=test_set,save_name=save_name,seed=seed)
