import numpy
import torch
import torch.nn as nn
from abstract.abstract_class_V import V
from torch.autograd import Variable
from general_util.pytorch_random import generate_gamma
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_class_point import point
from explicit.general_util import logsumexp_torch
from experiments.neural_net_experiments.gibbs_vs_joint_sampling.gibbs_vs_together_hyperparam import update_param_and_hyperparam_one_step
from abstract.mcmc_sampler import log_class
from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import test_error
from abstract.abstract_nuts_util import abstract_GNUTS
from general_util.pytorch_random import log_inv_gamma_density
from post_processing.ESS_nuts import diagnostics_stan
precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)

data_dict = get_data_dict("pima_indian")


class V_hierarchical_logistic_gibbs(V):
    def __init__(self,precision_type,gibbs=False):
        self.gibbs = gibbs
        super(V_hierarchical_logistic_gibbs, self).__init__(precision_type=precision_type)
    # def V_setup(self,y,X,lamb)
    def V_setup(self):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        data_dict = get_data_dict("pima_indian")
        X_np = data_dict["input"]
        y_np = data_dict["target"]
        self.dim = X_np.shape[1]
        num_ob = X_np.shape[0]
        self.num_ob = X_np.shape[0]

        self.beta = nn.Parameter(torch.zeros(self.dim),requires_grad=True)
        if self.gibbs:
            self.sigma2 = Variable(torch.zeros(1),requires_grad=False)
            self.list_hyperparam = [self.sigma2]
            self.list_param = [self.beta]
        else:
            self.log_sigma2 = nn.Parameter(torch.zeros(1),requires_grad=True)
            #self.sigma2 = Variable(self.log_sigma2.data,requires_grad=False)
        # sigma mapped to log space beecause we want it unconstrained
        # self.beta[self.dim] = log(sigma)
        #self.logsigma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.y = Variable(torch.from_numpy(y_np),requires_grad=False).type(precision_type)
        self.X = Variable(torch.from_numpy(X_np),requires_grad=False).type(precision_type)
        # parameter for hyperprior distribution

        self.lamb = 1
        return()

    def forward(self):
        if self.gibbs:
            print("sigma2 {}".format(self.sigma2))
        else:
            print("sigma2 {}".format(torch.exp(self.log_sigma2)))
        beta = self.beta
        if self.gibbs:
            sigma2 = self.sigma2
        else:
            sigma2 = torch.exp(self.log_sigma2)
        likelihood = torch.dot(beta, torch.mv(torch.t(self.X), self.y)) - \
                     torch.sum(logsumexp_torch(Variable(torch.zeros(self.num_ob)), torch.mv(self.X, beta)))
        prior = (-(beta*beta)/(sigma2)-torch.log(sigma2)).sum() * 0.5
        if not self.gibbs:
            prior += log_inv_gamma_density(x=sigma2, alpha=0.5, beta=0.5)
            prior += self.log_sigma2

        #hessian_term = -self.beta[self.dim-1]
        posterior = prior + likelihood
        out = -posterior
        return(out)
    def predict(self,test_samples):
        X = torch.from_numpy(test_samples)
        out = torch.zeros(X.shape[0],2)
        out[:,1] = (torch.sigmoid(torch.mv(X, self.beta.data)))
        out[:,0] = 1-out[:,1]
        return(out)
    def load_hyperparam(self,list_hyperparam):
        # input needs to be list of tensors
        for i in range(len(self.list_hyperparam)):
            self.list_hyperparam[i].data.copy_(list_hyperparam[i])
        return()

    def get_hyperparam(self):
        out = []
        for i in range(len(self.list_hyperparam)):
            out.append(self.list_hyperparam[i].data.clone())
        return(out)

    def update_hyperparam(self):
        alpha_tensor = torch.zeros(len(self.list_hyperparam))
        beta_tensor = torch.zeros(len(self.list_hyperparam))
        for i in range(len(self.list_hyperparam)):
            n = len(self.list_param[i].data.view(-1))
            norm = ((self.list_param[i].data)*(self.list_param[i].data)).sum()
            alpha_tensor[i] = n*0.5 + 0.5
            beta_tensor[i] = norm *0.5 + 0.5
        new_hyperparam_val = 1/generate_gamma(alpha=alpha_tensor,beta=beta_tensor)
        for i in range(len(self.list_hyperparam)):
            self.list_hyperparam[i].data[0] = new_hyperparam_val[i]
        return()
    def load_explicit_gradient(self):
        return()

    def load_explicit_H(self):
        # write down explicit hessian
        return()
    def load_explicit_dH(self):

        return()

    def load_explicit_diagH(self):

        return ()
    def load_explicit_graddiagH(self):

        return()



############################################################################################################################
#input_data = get_data_dict("pima_indian",standardize_predictor=True)

import os, numpy,torch
import dill as pickle
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from post_processing.ESS_nuts import ess_stan,diagnostics_stan
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics
#prior_dict = {"name":"horseshoe_3"}
#model_dict = {"num_units":10}
# hyperparameter has about same ess across seeds

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=4000,num_chains=4,num_cpu=4,thin=1,tune_l_per_chain=1000,
                                   warmup_per_chain=1100,is_float=False,isstore_to_disk=False,allow_restart=False,seed=58)

# input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],
#                "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict = {"v_fun":[V_hierarchical_logistic_gibbs],"epsilon":["dual"],"second_order":[False],"cov":["adapt"],"max_tree_depth":[8],
               "metric_name":["diag_e"],"dynamic":[True],"windowed":[False],"criterion":["gnuts"]}
# input_dict = {"v_fun":[v_generator],"epsilon":[0.1],"second_order":[False],"evolve_L":[10],
#               "metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}
ep_dual_metadata_argument = {"name":"epsilon","target":0.8,"gamma":0.05,"t_0":10,
                         "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
#
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=V_hierarchical_logistic_gibbs(precision_type="torch.DoubleTensor").get_model_dim())]
dual_args_list = [ep_dual_metadata_argument]
other_arguments = other_default_arguments()
#tune_settings_dict = tuning_settings([],[],[],[])
tune_settings_dict = tuning_settings(dual_args_list,[],adapt_cov_arguments,other_arguments)
tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

sampler1.start_sampling()

samples = sampler1.get_samples(permuted=False)

transformed_samples = samples
transformed_samples[:,:,7] = numpy.exp(samples[:,:,7])
out = diagnostics_stan(mcmc_samples_tensor=transformed_samples)

print(numpy.mean(transformed_samples[:,:,7]))
print(numpy.var(transformed_samples[:,:,7]))
print(out)