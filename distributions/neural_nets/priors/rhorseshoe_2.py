from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density,log_inv_gamma_density

# regularized horseshoe prior cp parametrization for the model weight
# cp parametrization for local lamb and global tau
# ncp parametrization for c^^2

class rhorseshoe_2(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1,slab_df=4,slab_scale=2):
        # assume prior c^2 ~ Inv-Gamma(alpha,beta) == Inv-Gamma(slab_df/2,slab_df * slab_scale*slab_scale/2)
        # slab_df =4 ,slab_scale =2 >>> alpha 2 , beta = 8
        self.global_scale = global_scale
        self.nu = nu
        self.slab_df = slab_df
        self.slab_scale = slab_scale
        self.name = name
        self.relevant_param_tuple = ("w","lamb","lamb_tilde","c","tau")
        self.setup_parameter(obj,shape)
        super(rhorseshoe_2, self).__init__()

    def get_val(self):
        return(self.w_obj)

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb2 = torch.exp(self.log_lamb2_obj)
        tau2 = torch.exp(self.log_tau2_obj)
        lamb_tilde2 = c * c * lamb2 / (c * c + tau2 * lamb2)
        lamb_tilde = torch.sqrt(lamb_tilde2)

        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            if name == "w":
                out = self.w_obj
            elif name =="tau":
                out = torch.sqrt(tau2)
            elif name =="c":
                out = c
            elif name == "lamb_tilde":
                out = lamb_tilde
            elif name == "lamb":
                out = torch.sqrt(lamb2)
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)




    def get_out(self):

        lamb2 = torch.exp(self.log_lamb2_obj)
        tau2 = torch.exp(self.log_tau2_obj)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde2 = c * c * lamb2  / (c * c + tau2 * lamb2)
        lamb_tilde = torch.sqrt(lamb_tilde2)
        tau = torch.sqrt(tau2)
        lamb2_out = log_student_t_density(x=lamb2, nu=1, mu=0, sigma=1) + self.log_lamb2_obj.sum()
        tau2_out = log_student_t_density(x=tau2, nu=self.nu, mu=0, sigma=self.global_scale) + self.log_tau2_obj.sum()
        c_r1_out = -(c_r1*c_r1).sum() * 0.5 + self.log_c_r1_obj
        c_alpha = self.slab_df / 2
        c_beta = self.slab_df * self.slab_scale * self.slab_scale / 2
        c_r2_out = log_inv_gamma_density(x=c_r2, alpha=c_alpha, beta=c_beta) + self.log_c_r2_obj.sum()

        w_out = -(self.w_obj*self.w_obj/(lamb_tilde*lamb_tilde*tau*tau)).sum()*0.5

        out = w_out + lamb2_out + tau2_out + c_r1_out + c_r2_out
        return(out)

    def setup_parameter(self,obj,name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_lamb2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_tau2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj,name+"_w_obj_"+self.name,self.w_obj)
        setattr(obj,name+"_lamb2_obj_"+self.name,self.log_lamb2_obj)
        setattr(obj,name+"_tau2_obj_"+self.name,self.log_tau2_obj)
        setattr(obj,name+ "_c_log_r1_obj_"+self.name, self.log_c_r1_obj)
        setattr(obj,name+ "_c_log_r2_obj_"+self.name, self.log_c_r2_obj)
        return()