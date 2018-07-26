from final_experiments.gibbs_vs_joint.setup import setup_gibbs_v_joint_experiment
from input_data.convert_data_to_dict import get_data_dict

seed = 12
input_data = get_data_dict("8x8mnist")
train_set = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,:],"target":input_data["target"][-500:]}


save_name = "gibbs_v_joint_convergence_results.npz"


num_units_list = [35]

setup_gibbs_v_joint_experiment(num_units_list=num_units_list,train_set=train_set,test_set=test_set,num_samples=1000,
                               save_name=save_name,seed=seed)

