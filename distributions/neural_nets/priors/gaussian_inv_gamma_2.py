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
        self.setup_parameter(obj, name, shape)
        # super(horseshoe_1, self).__init__()

    def get_val(self):
        sigma = torch.exp(self.log_sigma_obj)
        w_obj = self.z_obj * torch.sqrt(sigma) * self.global_scale
        return (w_obj)

    def get_out(self):
        sigma = torch.exp(self.log_sigma_obj)
        precision_out = log_inv_gamma_density(x=sigma,alpha=self.global_df*0.5,beta=self.global_scale*0.5) + self.log_sigma_obj.sum()
        z_out = -(self.z_obj * self.z_obj * sigma).sum() * 0.5
        out = z_out + precision_out
        return (out)

    def setup_parameter(self, obj, name, shape):
        self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_sigma_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, "z_obj", self.z_obj)
        setattr(obj, "log_sigma_obj", self.log_sigma_obj)
        return ()