import time
import numpy
start_time = time.time()
run_time_list = []
experiments_name_list = []
import final_experiments.adapt_cov.run_experiment
experiments_name_list.append("adapt_cov")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.effect_of_scaling.run_experiment
experiments_name_list.append("effect_of_scaling")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.effects_of_prior.run_experiment
experiments_name_list.append("effects_of_prior")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.ensemble_vs_mcmc.run_experiment
experiments_name_list.append("ensemble_vs_mcmc")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.float_v_double.run_experiment
experiments_name_list.append("float_vs_double")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.gibbs_vs_joint.run_experiment
experiments_name_list.append("gibbs_vs_joint")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.hmc_v_gnuts.run_experiment
experiments_name_list.append("hmc_vs_gnuts")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.num_layers.run_experiment
experiments_name_list.append("num_layers")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.sghmc_vs_fulldata.run_experiment
experiments_name_list.append("sghmc_vs_fulldata")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.waic_test_error.run_experiment
experiments_name_list.append("waic_test")
run_time_list.append(time.time()-start_time)
start_time = time.time()
import final_experiments.xhmc_gnuts.run_experiment
experiments_name_list.append("xhmc_gnuts")
run_time_list.append(time.time()-start_time)

total_time = sum(run_time_list)
print(total_time)

out = {"total_time":total_time,"run_time_list":run_time_list,"experiments_name_list":experiments_name_list}
numpy.savez("run_time.npz",**out)