from distributions.neural_nets.priors.base_class import base_prior
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density
# horseshoe prior ncp parametrization for the model weight
# cp parametrization for local lamb and global tau
class horseshoe1(base_prior):
    def __init__(self):
        super(horseshoe1, self).__init__()

    def get_val(self):
        local_lamb = torch.exp(self.log_local_lamb_obj)
        global_tau = torch.exp(self.log_global_tau_obj)
        w_obj = self.z_obj * local_lamb*global_tau
        return(w_obj)

    def get_out(self):
        z_out = (self.z_obj*self.z_obj).sum()*0.5
        local_lamb = torch.exp(self.log_local_lamb_obj)
        global_tau = torch.exp(self.log_global_tau_obj)
        local_lamb_out = log_student_t_density(x=local_lamb,nu=1,mu=0,sigma=1) + self.log_local_lamb_obj.sum()
        global_tau_out = log_student_t_density(x=global_tau,nu=1,mu=0,sigma=1) + self.log_global_tau_obj.sum()
        out = z_out + local_lamb_out + global_tau_out
        return(out)

    def setup_parameter(self,obj, name, shape):
        self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_lamb_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_global_tau_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj,"z_obj",self.z_obj)
        setattr(obj,"local_lamb_obj",self.log_local_lamb_obj)
        setattr(obj,"global_tau_obj",self.log_global_tau_obj)
        return()