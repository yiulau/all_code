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
    def __init__(self,precision_type,gibbs):
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


v_obj = V_hierarchical_logistic_gibbs(precision_type="torch.DoubleTensor",gibbs=True)
metric_obj = metric(name="unit_e",V_instance=v_obj)
Ham = Hamiltonian(v_obj,metric_obj)

init_q_point = point(V=v_obj)
init_hyperparam = [torch.abs(torch.randn(1))+3]
log_obj = log_class()

#print(init_q_point.flattened_tensor)

num_samples = 1000
dim = len(init_q_point.flattened_tensor)
mcmc_samples_weight = torch.zeros(num_samples,dim)
mcmc_samples_hyper = torch.zeros(num_samples)
for i in range(num_samples):
    outq,out_hyperparam = update_param_and_hyperparam_one_step(init_q_point,init_hyperparam,Ham,0.1,10,log_obj)
    init_q_point.flattened_tensor.copy_(outq.flattened_tensor)
    init_q_point.load_flatten()
    init_hyperparam = out_hyperparam
    mcmc_samples_weight[i,:] = outq.flattened_tensor.clone()
    mcmc_samples_hyper[i:i+1] = out_hyperparam[0].clone()
mcmc_samples_weight = mcmc_samples_weight.numpy()
mcmc_samples_hyper = mcmc_samples_hyper.numpy()
print(mcmc_samples_weight.shape)

print("sigma diagnostics gibbs")
print(numpy.mean(mcmc_samples_hyper))
print(numpy.var(mcmc_samples_hyper))

print("weight diagnostics gibbs")
print(numpy.mean(mcmc_samples_weight,axis=0))


v_obj2 = V_hierarchical_logistic_gibbs(precision_type="torch.DoubleTensor",gibbs=False)
metric_obj = metric(name="unit_e",V_instance=v_obj2)
Ham = Hamiltonian(v_obj2,metric_obj)

q_point = point(V=Ham.V)
inputq = torch.randn(len(q_point.flattened_tensor))
print(len(inputq))

q_point.flattened_tensor.copy_(inputq)
q_point.load_flatten()

chain_l=1000
store_samples = torch.zeros(1,chain_l,len(inputq))
store_divergent = torch.zeros(chain_l)
for i in range(chain_l):
    out = abstract_GNUTS(init_q=q_point,epsilon=0.1,Ham=Ham,max_tree_depth=10)
    store_samples[0,i,:] = out[0].flattened_tensor.clone()
    q_point = out[0]
    store_divergent[i] = out[6]

print("num divergent {}".format(store_divergent.sum()))

store_samples = store_samples[0,:,:].numpy()
print(store_samples.shape)
print("diagnostics full hmc")
print(numpy.mean(store_samples,axis=0))
print(numpy.var(store_samples,axis=0))

print("diagnostics full hmc sigma2 ")
print(numpy.mean(numpy.exp(store_samples[:,7])))
print(numpy.var(numpy.exp(store_samples[:,7])))
#print(store_samples)
sigma2_tensor = numpy.zeros((1,store_samples.shape[0],store_samples.shape[1]))
sigma2_tensor[0,:,:] = numpy.exp(store_samples)

print(diagnostics_stan(mcmc_samples_tensor=sigma2_tensor))

print("sigma diagnostics gibbs")
print(numpy.mean(mcmc_samples_hyper))
print(numpy.var(mcmc_samples_hyper))
sigma2_tensor = numpy.zeros((1,len(mcmc_samples_hyper),1))
sigma2_tensor[0,:,0] = mcmc_samples_hyper
print(diagnostics_stan(mcmc_samples_tensor=sigma2_tensor))
print("weight diagnostics gibbs")

print(numpy.mean(mcmc_samples_weight,axis=0))
weight_tensor = numpy.zeros((1,mcmc_samples_weight.shape[0],mcmc_samples_weight.shape[1]))
weight_tensor[0,:,:] = mcmc_samples_weight
print(diagnostics_stan(mcmc_samples_tensor=weight_tensor))
