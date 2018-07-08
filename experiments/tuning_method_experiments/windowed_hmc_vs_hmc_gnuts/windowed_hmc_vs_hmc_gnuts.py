from experiments.tuning_method_experiments.windowed_hmc_vs_hmc_gnuts.util import choose_optimal_L,min_ess_gnuts,convert_results_to_np

import numpy, os
v_fun_list = []
# ep_list list of conservatively chosen epsilon for each model
# compare models : logit,mvn,nn
# compare ess performance of hmc and windowed hmc with optimally chosen L to gnuts
# unit_e
ep_list = []
#num_repeats = 50
num_of_L = 2

# for i in range(num_repeats):
L_list = [round(a.item()) for a in list(numpy.linspace(5,1024,num_of_L))]
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




converted_results = convert_results_to_np(results_list)

save_address = "temp.npz"
numpy.savez(save_address,**converted_results)











