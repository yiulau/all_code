from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# gaussian-inv_gamma prior ncp parametrization for the model weight
# only one sigma shared by all weights
class gaussian_inv_gamma_2(base_prior_new):
    def __init__(self, obj, name, shape,global_scale=1,global_df=1):
        self.global_df = global_df
        self.global_scale = global_scale
        self.name = name
        self.relevant_param_tuple = ("w","sigma2")
        self.setup_parameter(obj, name, shape)

    def get_val(self):
        sigma2 = torch.exp(self.log_sigma2_obj)
        w_obj = self.z_obj * torch.sqrt(sigma2) * self.global_scale
        return (w_obj)

    def get_out(self):
        sigma2 = torch.exp(self.log_sigma2_obj)
        sigma2_out = log_inv_gamma_density(x=sigma2,alpha=self.global_df*0.5,beta=self.global_df*0.5) + self.log_sigma2_obj.sum()
        z_out = -(self.z_obj * self.z_obj).sum() * 0.5
        out = z_out + sigma2_out
        return (out)

    def setup_parameter(self, obj, name, shape):
        self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_sigma2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, name+"_z_obj", self.z_obj)
        setattr(obj, name+"_log_sigma2_obj", self.log_sigma2_obj)
        return ()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple
        sigma2 = torch.exp(self.log_sigma2_obj)
        w_obj = self.z_obj * torch.sqrt(sigma2)
        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            if name == "w":
                out = w_obj
            elif name =="sigma2":
                out = sigma2
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)