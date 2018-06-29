from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# horseshoe prior cp parametrization for the model weight
# ncp parametrization for local lamb and global tau
class horseshoe_4(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1):
        self.global_scale = global_scale
        self.nu = nu
        self.name = name
        self.relevant_param_tuple = ("w", "lamb", "tau")
        self.setup_parameter(obj, name, shape)
        super(horseshoe_4, self).__init__()

    def get_val(self):

        return (self.w_obj)

    def get_out(self):

        local_r1 = torch.exp(self.log_local_r1_obj)
        local_r2 = torch.exp(self.log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)

        local_r1_out = -(local_r1 * local_r1).sum() * 0.5 + self.log_local_r1_obj.sum()
        global_r1_out = -(global_r1 * global_r1).sum() * 0.5 + self.log_global_r1_obj.sum()


        local_r2_out = log_inv_gamma_density(x=local_r2, alpha=0.5, beta=0.5) + self.log_local_r2_obj.sum()
        global_r2_out = log_inv_gamma_density(x=global_r2, alpha=0.5*self.nu,beta=0.5*self.nu) + self.log_global_r2_obj.sum()
        tau = global_r1 * torch.sqrt(global_r2)*self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)

        w_out = -(self.w_obj * self.w_obj /(tau*tau*lamb*lamb)).sum() * 0.5 -(torch.log(tau*tau*lamb*lamb)).sum()*0.5
        out = w_out + local_r2_out + global_r2_out + local_r1_out + global_r1_out
        return (out)

    def get_unit_scales(self):
        # assume shape = [num_units,num_in_units]
        assert len(self.w_obj.shape) ==  2
        local_r1 = torch.exp(self.log_local_r1_obj)
        local_r2 = torch.exp(self.log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)
        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)

        out = torch.sqrt((tau * tau * lamb * lamb).sum(dim=1))
        return(out)

    def setup_parameter(self, obj, name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_r1_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_r2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_global_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_global_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj, name+"_w_obj", self.w_obj)
        setattr(obj, name+"_log_local_r1_obj", self.log_local_r1_obj)
        setattr(obj, name+"_log_local_r2_obj", self.log_local_r2_obj)
        setattr(obj, name+"_log_global_r1_obj", self.log_global_r1_obj)
        setattr(obj, name+"_log_global_r2_obj", self.log_global_r2_obj)
        return ()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple
        local_r2 = torch.exp(self.log_local_r2_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)
        local_r1 = torch.exp(self.log_local_r1_obj)
        global_r1 = torch.exp(self.log_global_r2_obj)
        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)

        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = self.w_obj
            elif name =="tau":
                out = tau
            elif name == "lamb":
                out = lamb
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)