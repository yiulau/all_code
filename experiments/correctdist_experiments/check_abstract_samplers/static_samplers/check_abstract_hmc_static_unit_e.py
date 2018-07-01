import pickle,os,torch
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class
from input_data.convert_data_to_dict import get_data_dict
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from abstract.util import wrap_V_class_with_input_data

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=1000,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=500,is_float=False,isstore_to_disk=False,allow_restart=False)
input_data = get_data_dict("pima_indian")
V_pima_indian_logit = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain.pkl"
correct = pickle.load(open(address, 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()


input_dict = {"v_fun":[V_pima_indian_logit],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}
input_dict2 = {"v_fun":[V_pima_indian_logit],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[True],"criterion":[None]}

tune_settings_dict = tuning_settings([],[],[],[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()
tune_dict2  = tuneinput_class(input_dict2).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)
sampler2 = mcmc_sampler(tune_dict=tune_dict2,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

sampler1.start_sampling()
sampler2.start_sampling()


mcmc_samples1 = sampler1.get_samples(permuted=False)

out = check_mean_var_stan(mcmc_samples=mcmc_samples1,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)


mcmc_samples2 = sampler2.get_samples(permuted=False)


out = check_mean_var_stan(mcmc_samples=mcmc_samples2,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)

