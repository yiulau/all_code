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
from experiments.neural_net_experiments.gibbs_vs_together_hyperparam import update_param_and_hyperparam_one_step
from abstract.mcmc_sampler import log_class
from input_data.convert_data_to_dict import get_data_dict
precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)

data_dict = get_data_dict("pima_indian")


class V_hierarchical_logistic_gibbs(V):
    #def __init__(self):
    #    super(V_test_abstract, self).__init__()
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
        self.sigma = Variable(torch.zeros(1),requires_grad=False)
        # sigma mapped to log space beecause we want it unconstrained
        # self.beta[self.dim] = log(sigma)
        #self.logsigma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.y = Variable(torch.from_numpy(y_np),requires_grad=False).type(precision_type)
        self.X = Variable(torch.from_numpy(X_np),requires_grad=False).type(precision_type)
        # parameter for hyperprior distribution
        self.list_hyperparam = [self.sigma]
        self.list_param = [self.beta]
        self.lamb = 1
        return()

    def forward(self):
        beta = self.beta
        sigma = self.sigma
        likelihood = torch.dot(beta, torch.mv(torch.t(self.X), self.y)) - \
                     torch.sum(logsumexp_torch(Variable(torch.zeros(self.num_ob)), torch.mv(self.X, beta)))
        prior = -torch.dot(beta, beta)/(sigma*sigma) * 0.5 - torch.log(sigma*sigma)*0.5

        #hessian_term = -self.beta[self.dim-1]
        posterior = prior + likelihood
        out = -posterior
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
            alpha_tensor[i] = n*0.5 + 1
            beta_tensor[i] = norm *0.5 + 1
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


v_obj = V_hierarchical_logistic_gibbs()
metric = metric(name="unit_e",V_instance=v_obj)
Ham = Hamiltonian(v_obj,metric)

init_q_point = point(V=v_obj)
init_hyperparam = [torch.abs(torch.randn(1))]
log_obj = log_class()

print(init_q_point.flattened_tensor)

out = update_param_and_hyperparam_one_step(init_q_point,init_hyperparam,Ham,0.1,10,log_obj)

print(out.flattened_tensor)






