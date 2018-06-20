#from distributions.logistic_regressions.logistic_horseshoe import class_generator
from distributions.linear_regressions.linear_regression_horseshoe import V_linear_regression_hs
from abstract.util import wrap_V_class_with_input_data
from distributions.neural_nets.priors.prior_util import prior_generator
import os, numpy,torch,pickle
import pandas as pd

# abs_address = os.environ["PYTHONPATH"] + "/input_data/pima_india.csv"
# df = pd.read_csv(abs_address, header=0, sep=" ")
# # print(df)
# dfm = df.values
# # print(dfm)
# # print(dfm.shape)
# y_np = dfm[:, 8]
# y_np = y_np.astype(numpy.int64)
# X_np = dfm[:, 1:8]

non_zero_num_p = 10
full_p = 50
num_samples = 60
X_np = numpy.random.randn(num_samples,full_p)*5
true_beta = numpy.zeros(full_p)
true_beta[:non_zero_num_p] = numpy.random.randn(non_zero_num_p)*5
y_np = X_np.dot(true_beta) + numpy.random.randn(num_samples)
input_data = {"input":X_np,"target":y_np}


#prior_generator_fun = prior_generator("horseshoe_1")
#v_generator = class_generator(input_data,prior_generator_fun)
v_generator =wrap_V_class_with_input_data(class_constructor=V_linear_regression_hs,input_data=input_data)

from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from experiments.experiment_obj import tuneinput_class
from distributions.two_d_normal import V_2dnormal
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from post_processing.ESS_nuts import ess_stan
seedid = 33350
numpy.random.seed(seedid)
torch.manual_seed(seedid)
mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=1000,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=100,is_float=False,isstore_to_disk=False,allow_restart=False)

# input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
#               "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict = {"v_fun":[v_generator],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

tune_settings_dict = tuning_settings([],[],[],[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

out = sampler1.start_sampling()



mcmc_samples = sampler1.get_samples(permuted=False)
#print(mcmc_samples.shape)
#print("mcmc mean {}".format(numpy.mean(mcmc_samples,axis=0)))
print(ess_stan(mcmc_samples))
#from python2R.ess_rpy2 import ess_repy2
mcmc_samples_mixed = sampler1.get_samples(permuted=False)
#print(ess_repy2(mcmc_samples_mixed))
exit()
#print(numpy.cov(mcmc_samples,rowvar=False))

address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain.pkl"
correct = pickle.load(open(address, 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()
print("exact mean {}".format(correct_mean))

output = check_mean_var_stan(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = output["mcmc_mean"],output["mcmc_Cov"]
pc_mean,pc_cov = output["pc_of_mean"],output["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)

#print(len(v_obj.list_tensor))

#print(list(v_obj.named_parameters()))

