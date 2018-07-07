import os,torch
import dill as pickle
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class
from input_data.convert_data_to_dict import get_data_dict
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from abstract.util import wrap_V_class_with_input_data

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=2000,num_chains=4,num_cpu=4,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False,allow_restart=False)
input_data = get_data_dict("pima_indian")
V_pima_indian_logit = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain.pkl"
correct = pickle.load(open(address, 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()

ideal_cov = (torch.from_numpy(correct_cov).type("torch.DoubleTensor")).inverse()



input_dict3 = {"v_fun":[V_pima_indian_logit],"epsilon":["dual"],"second_order":[False],"cov":[ideal_cov],
              "evolve_t":[1.5],"metric_name":["dense_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}
ep_dual_metadata_argument = {"name":"epsilon","target":0.8,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()
tune_dict  = tuneinput_class(input_dict3).singleton_tune_dict()
tune_settings_dict = tuning_settings(dual_args_list,[],[],other_arguments)


tune_dict3  = tuneinput_class(input_dict3).singleton_tune_dict()

sampler3 = mcmc_sampler(tune_dict=tune_dict3,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

store_name = 'standard_sampler.pkl'
sampled = False
if sampled:
    sampler3 = pickle.load(open(store_name, 'rb'))
else:
    sampler3.start_sampling()
    with open(store_name, 'wb') as f:
        pickle.dump(sampler3, f)





mcmc_samples3= sampler3.get_samples(permuted=False)

out = check_mean_var_stan(mcmc_samples=mcmc_samples3,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)



