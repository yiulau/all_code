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
        self.setup_parameter(obj,name,shape)
        super(rhorseshoe_2, self).__init__()

    def get_val(self):
        return(self.w_obj)

    def get_out(self):

        lamb = torch.exp(self.log_lamb_obj)
        tau = torch.exp(self.log_tau_obj)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb)

        lamb_out = log_student_t_density(x=lamb, nu=1, mu=0, sigma=1) + self.log_lamb_obj.sum()
        tau_out = log_student_t_density(x=tau, nu=self.nu, mu=0, sigma=self.global_scale) + self.log_tau_obj.sum()
        c_r1_out = -(c_r1*c_r1).sum() * 0.5 + self.log_c_r1_obj
        c_alpha = self.slab_df / 2
        c_beta = self.slab_df * self.slab_scale * self.slab_scale / 2
        c_r2_out = log_inv_gamma_density(x=c_r2, alpha=c_alpha, beta=c_beta) + self.log_c_r2_obj.sum()

        w_out = -(self.w_obj*self.w_obj/(lamb_tilde*lamb_tilde*tau*tau)).sum()*0.5

        out = w_out + lamb_out + tau_out + c_r1_out + c_r2_out
        return(out)

    def setup_parameter(self,obj, name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_lamb_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_tau_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj,"w_obj",self.w_obj)
        setattr(obj,"lamb_obj",self.log_lamb_obj)
        setattr(obj,"tau_obj",self.log_tau_obj)
        setattr(obj, "c_log_r1_obj", self.log_c_r1_obj)
        setattr(obj, "c_log_r2_obj", self.log_c_r2_obj)
        return()