import time
import numpy
start_time = time.time()
import final_experiments.adapt_cov.run_experiment
import final_experiments.effect_of_scaling.run_experiment
import final_experiments.effects_of_prior.run_experiment
import final_experiments.ensemble_vs_mcmc.run_experiment
import final_experiments.float_v_double.run_experiment
import final_experiments.gibbs_vs_joint.run_experiment
import final_experiments.hmc_v_gnuts.run_experiment
import final_experiments.num_layers.run_experiment
import final_experiments.sghmc_vs_fulldata.run_experiment
import final_experiments.waic_test_error.run_experiment
import final_experiments.xhmc_gnuts.run_experiment

total_time = time.time() - start_time()

numpy.savez("run_time.npz",total_time)