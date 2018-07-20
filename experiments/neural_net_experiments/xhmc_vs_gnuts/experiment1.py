# fix prior standard normal
# compare min ess for different value of xhmc_delta
from distributions.neural_nets.fc_V_model_1 import V_fc_model_1
from input_data.convert_data_to_dict import get_data_dict
from experiments.neural_net_experiments.xhmc_vs_gnuts.util import run_nn_experiment

xhmc_delta_list = [0.01,0.05,0.1]
input_data = get_data_dict("8x8mnist")
input_data = {"input":input_data["input"][:500,],"target":input_data["target"][:500]}
test_set = {"input":input_data["input"][-500:,],"target":input_data["target"][-500:]}
v_fun = V_fc_model_1

out = run_nn_experiment(xhmc_delta_list=xhmc_delta_list,input_data=input_data,v_fun=v_fun,test_set=test_set,type_problem="classification")

