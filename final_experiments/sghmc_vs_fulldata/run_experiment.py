from final_experiments.sghmc_vs_fulldata.setup import setup_sghmc_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:700,:],"target":input_data["target"][:700]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "sghmc_results.npz"


ep_list = [1e-1,1e-2,1e-3,1e-4]
L_list = [10,50,100,200]
eta_list = [1e-1,1e-2,1e-3,1e-4]

setup_sghmc_experiment(ep_list=ep_list,L_list=L_list,eta_list=eta_list,train_set=train_set,test_set=test_set,
                               save_name=save_name,seed=seed)



