from experiments.neural_net_experiments.effect_of_scaling.util import run_nn_experiment
from input_data.convert_data_to_dict import get_data_dict
from experiments.neural_net_experiments.effect_of_scaling.scaled_model import V_fc_scaled_model
from experiments.neural_net_experiments.effect_of_scaling.unscaled_model import V_fc_unscaled_model
input_data = get_data_dict("8x8mnist")
test_set = {"input":input_data["input"][-500:,],"target":input_data["target"][-500:]}
train_set = {"input": input_data["input"][:500, ], "target": input_data["target"][:500]}

out_scaled = run_nn_experiment(list_num_units=[10,50,100],input_data=train_set,v_fun=V_fc_scaled_model,test_set=test_set,type_problem="classification")

out_unscaled = run_nn_experiment(list_num_units=[10,50,100],input_data=train_set,v_fun=V_fc_unscaled_model,test_set=test_set,type_problem="classification")

