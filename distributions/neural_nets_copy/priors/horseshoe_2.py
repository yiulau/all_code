from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density
# horseshoe prior cp parametrization for the model weight
# cp parametrization for local lamb and global tau
class horseshoe_2(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1):
        self.global_scale = global_scale
        self.nu = nu
        self.setup_parameter(obj,name,shape)
        #super(horseshoe_2, self).__init__()

    def get_val(self):
        return(self.w_obj)

    def get_out(self):

        lamb = torch.exp(self.log_lamb_obj)
        tau = torch.exp(self.log_tau_obj)
        lamb_out = log_student_t_density(x=lamb,nu=1,mu=0,sigma=1) + self.log_lamb_obj.sum()
        tau_out = log_student_t_density(x=tau,nu=self.nu,mu=0,sigma=self.global_scale) + self.log_tau_obj.sum()
        w_out = -(self.w_obj*self.w_obj/(lamb*lamb*tau*tau)).sum()*0.5

        out = w_out + lamb_out + tau_out
        return(out)

    def setup_parameter(self,obj, name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_lamb_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_tau_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj,"w_obj",self.w_obj)
        setattr(obj,"lamb_obj",self.log_lamb_obj)
        setattr(obj,"tau_obj",self.log_tau_obj)
        return()