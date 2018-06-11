from distributions.neural_nets.priors.base_class import base_prior
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# horseshoe prior cp parametrization for the model weight
# ncp parametrization for local lamb and global tau
class horseshoe_4(base_prior):
    def __init__(self):
        super(horseshoe_4, self).__init__()

    def get_val(self):

        return (self.w_obj)

    def get_out(self):

        local_r1_out = -(self.local_r1_obj * self.local_r1_obj).sum() * 0.5
        global_r1_out = -(self.global_r1_obj * self.global_r1_obj).sum() * 0.5
        local_r2 = torch.exp(self.log_local_lamb_obj)
        global_r2 = torch.exp(self.log_global_tau_obj)

        local_r2_out = log_inv_gamma_density(x=local_r2, nu=1, mu=0, sigma=1) + self.log_local_lamb_obj.sum()
        global_r2_out = log_inv_gamma_density(x=global_r2, nu=1, mu=0, sigma=1) + self.log_global_tau_obj.sum()
        tau = self.global_r1_obj * torch.sqrt(global_r2)
        lamb = self.local_r1_obj * torch.sqrt(local_r2)

        w_out = -(self.w_obj * self.w_obj /(tau*tau*lamb*lamb)).sum() * 0.5
        out = w_out + local_r2_out + global_r2_out + local_r1_out + global_r1_out
        return (out)

    def setup_parameter(self, obj, name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.local_r1_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_r2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.global_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_global_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj, "w_obj", self.w_obj)
        setattr(obj, "local_r1_obj", self.local_r1_obj)
        setattr(obj, "log_local_r2_obj", self.log_local_r2_obj)
        setattr(obj, "global_r1_obj", self.global_r1_obj)
        setattr(obj, "log_global_r2_obj", self.log_global_r2_obj)
        return ()