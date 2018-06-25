from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density,log_inv_gamma_density
# regularized horseshoe prior ncp parametrization for the model weight
# ncp parametrization for local lamb and global tau
# ncp parametrization for c^^2

class rhorseshoe_3(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1,slab_df=4,slab_scale=2):
        # assume prior c^2 ~ Inv-Gamma(alpha,beta) == Inv-Gamma(slab_df/2,slab_df * slab_scale*slab_scale/2)
        # slab_df =4 ,slab_scale =2 >>> alpha 2 , beta = 8
        self.global_scale = global_scale
        self.nu = nu
        self.slab_df = slab_df
        self.slab_scale = slab_scale
        self.name = name
        self.setup_parameter(obj,shape)
        super(rhorseshoe_3, self).__init__()

    def get_val(self):
        local_r2 = torch.exp(self.log_local_r2_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)
        local_r1 = torch.exp(self.log_local_r1_obj)
        global_r1 = torch.exp(self.log_global_r2_obj)
        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = torch.sqrt(c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb))
        w_obj = self.z_obj * lamb_tilde * tau
        return(w_obj)

    def get_out(self):
        local_r1 = torch.exp(self.log_local_r1_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        local_r2 = torch.exp(self.log_local_r2_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)

        z_out = -(self.z_obj*self.z_obj).sum()*0.5
        local_r1_out = -(local_r1*local_r1).sum()*0.5 + self.log_local_r1_obj.sum()
        global_r1_out = -(global_r1*global_r1).sum()*0.5 + self.log_global_r1_obj.sum()

        local_r2_out = log_inv_gamma_density(x=local_r2,alpha=0.5,beta=0.5) + self.log_local_r2_obj.sum()
        global_r2_out = log_inv_gamma_density(x=global_r2,alpha=0.5*self.nu,beta=0.5*self.nu) + self.log_global_r2_obj.sum()

        c_r1_out = -(c_r1 * c_r1).sum() * 0.5 + self.log_c_r1_obj
        c_alpha = self.slab_df / 2
        c_beta = self.slab_df * self.slab_scale * self.slab_scale / 2
        c_r2_out = log_inv_gamma_density(x=c_r2, alpha=c_alpha, beta=c_beta) + self.log_c_r2_obj.sum()

        out = z_out + local_r2_out + global_r2_out + local_r1_out + global_r1_out + c_r1_out + c_r2_out
        return(out)

    def setup_parameter(self,obj, shape):
        self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_r1_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_local_r2_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
        self.log_global_r1_obj = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.log_global_r2_obj = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.log_c_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj,"z_obj_"+self.name,self.z_obj)
        setattr(obj,"log_local_r1_obj_"+self.name,self.log_local_r1_obj)
        setattr(obj,"log_local_r2_obj_"+self.name,self.log_local_r2_obj)
        setattr(obj,"log_global_r1_obj_"+self.name,self.log_global_r1_obj)
        setattr(obj,"log_global_r2_obj_"+self.name,self.log_global_r2_obj)
        setattr(obj, "c_log_r1_obj_"+self.name, self.log_c_r1_obj)
        setattr(obj, "c_log_r2_obj_"+self.name, self.log_c_r2_obj)
        return()

