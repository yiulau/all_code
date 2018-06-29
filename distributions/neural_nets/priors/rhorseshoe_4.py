from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density

#
# regularized horseshoe prior cp parametrization for the model weight
# ncp parametrization for local lamb and global tau
# ncp parametrization for c^^2

class rhorseshoe_4(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1,slab_df=4,slab_scale=2):
        # assume prior c^2 ~ Inv-Gamma(alpha,beta) == Inv-Gamma(slab_df/2,slab_df * slab_scale*slab_scale/2)
        # slab_df =4 ,slab_scale =2 >>> alpha 2 , beta = 8
        self.global_scale = global_scale
        self.nu = nu
        self.slab_df = slab_df
        self.slab_scale = slab_scale
        self.name = name
        self.relevant_param_tuple = ("w","lamb","lamb_tilde","c","tau")

        self.setup_parameter(obj, name,shape)
        super(rhorseshoe_4, self).__init__()

    def get_val(self):
        return (self.w_obj)

    def get_out(self):

        local_r1 = torch.exp(self.log_local_r1_obj)
        local_r2 = torch.exp(self.log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)
        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = torch.sqrt(c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb))
        local_r1_out = -(local_r1 * local_r1).sum() * 0.5 + self.log_local_r1_obj.sum()
        global_r1_out = -(global_r1 * global_r1).sum() * 0.5 + self.log_global_r1_obj.sum()
        local_r2_out = log_inv_gamma_density(x=local_r2, alpha=0.5, beta=0.5) + self.log_local_r2_obj.sum()
        global_r2_out = log_inv_gamma_density(x=global_r2, alpha=0.5*self.nu,beta=0.5*self.nu) + self.log_global_r2_obj.sum()
        w_out = -(self.w_obj * self.w_obj /(tau*tau*lamb_tilde*lamb_tilde)).sum() * 0.5 -(torch.log(tau*tau*lamb_tilde*lamb_tilde)).sum()*0.5
        c_r1_out = -(c_r1 * c_r1).sum() * 0.5 + self.log_c_r1_obj
        c_alpha = self.slab_df / 2
        c_beta = self.slab_df * self.slab_scale * self.slab_scale / 2
        c_r2_out = log_inv_gamma_density(x=c_r2, alpha=c_alpha, beta=c_beta) + self.log_c_r2_obj.sum()
        out = w_out + local_r2_out + global_r2_out + local_r1_out + global_r1_out + c_r1_out + c_r2_out
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
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb)
        out = torch.sqrt((tau * tau * lamb_tilde * lamb_tilde).sum(dim=1))
        return(out)

    def setup_parameter(self, obj,name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_r1_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_r2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_global_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_global_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj, name+"_w_obj", self.w_obj)
        setattr(obj, name+"_log_local_r1_obj", self.log_local_r1_obj)
        setattr(obj, name+"_log_local_r2_obj", self.log_local_r2_obj)
        setattr(obj, name+"_log_global_r1_obj", self.log_global_r1_obj)
        setattr(obj, name+"_log_global_r2_obj", self.log_global_r2_obj)
        setattr(obj, name+"_c_log_r1_obj", self.log_c_r1_obj)
        setattr(obj, name+"_c_log_r2_obj", self.log_c_r2_obj)
        return ()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple

        local_r1 = torch.exp(self.log_local_r1_obj)
        local_r2 = torch.exp(self.log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)
        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = torch.sqrt(c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb))

        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = self.w_obj
            elif name =="tau":
                out = tau
            elif name == "lamb":
                out = lamb
            elif name=="lamb_tilde":
                out = lamb_tilde
            elif name=="c":
                out = c
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)