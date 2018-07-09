from experiments.neural_net_experiments.effect_of_scaling.util import run_nn_experiment
from input_data.convert_data_to_dict import get_data_dict
from experiments.neural_net_experiments.effect_of_scaling.scaled_model import V_fc_scaled_model
from experiments.neural_net_experiments.effect_of_scaling.unscaled_model import V_fc_unscaled_model
input_data = get_data_dict("pima_indian")


out_scaled = run_nn_experiment(list_num_units=[10,50,100],input_data=input_data,v_fun=V_fc_scaled_model,test_set=input_data,type_problem="classification")

out_unscaled = run_nn_experiment(list_num_units=[10,50,100],input_data=input_data,v_fun=V_fc_unscaled_model,test_set=input_data,type_problem="classification")

