from experiments.tuning_method_experiments.windowed_hmc_vs_hmc_gnuts.util import choose_optimal_L,min_ess_gnuts,convert_results_to_np
from experiments.experiment_util import wishart_for_cov
from distributions.mvn import V_mvn
import numpy, os
from abstract.util import wrap_V_class_with_input_data
input_data = {"input":wishart_for_cov(dim=50)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)

v_fun_list = [V_mvn1]
# ep_list list of conservatively chosen epsilon for each model
# compare models : logit,mvn,nn
# compare ess performance of hmc and windowed hmc with optimally chosen L to gnuts
# unit_e
ep_list = [0.05]
#num_repeats = 50
num_of_L = 3

# for i in range(num_repeats):
L_list = [round(a.item()) for a in list(numpy.linspace(5,100,num_of_L))]
results_list = [None]*len(v_fun_list)
for j in range(len(v_fun_list)):
    result_on_problem = {"hmc": None, "windowed_hmc": None, "gnuts": None}
    ep = ep_list[j]
    v_fun = v_fun_list[j]
    out_hmc = choose_optimal_L(v_fun=v_fun,fixed_ep=ep,L_list=L_list,windowed=False)
    out_windowed_hmc = choose_optimal_L(v_fun=v_fun,fixed_ep=ep,L_list=L_list,windowed=True)
    out_gnuts = min_ess_gnuts(v_fun=v_fun,ep=ep)
    result_on_problem.update({"hmc":out_hmc,"windowed_hmc":out_windowed_hmc,"gnuts":out_gnuts})
    results_list[j] = result_on_problem


print(results_list)
exit()
converted_results = convert_results_to_np(results_list)

save_address = "temp.npz"
numpy.savez(save_address,**converted_results)











