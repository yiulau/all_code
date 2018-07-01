# test if softabs is more sensible to initialization than unit_e
# test at diffrent levels of alpha
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from experiments.experiment_obj import experiment,experiment_setting_dict
num_per_model = 100
from experiments.experiment_obj import tuneinput_class

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=10000,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=1000,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False,True],"alpha":[1e6],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_obj = tuneinput_class(input_dict)

experiment_setting_dict = experiment_setting_dict(chain_length=10,num_repeat=num_per_model)
experiment_obj = experiment(input_object=input_obj,experiment_setting=experiment_setting_dict)

experiment_obj.run()

