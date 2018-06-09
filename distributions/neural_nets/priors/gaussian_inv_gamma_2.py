from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# gaussian-inv_gamma prior ncp parametrization for the model weight
#
class gaussian_inv_gamma_1(base_prior_new):
    def __init__(self, obj, name, shape):
        self.setup_parameter(obj, name, shape)
        # super(horseshoe_1, self).__init__()

    def get_val(self):
        precision = torch.exp(self.log_precision_obj)
        w_obj = self.z_obj/torch.sqrt(precision)
        return (self.w_obj)

    def get_out(self):
        precision = torch.exp(self.log_precision_obj)
        precision_out = log_inv_gamma_density(x=precision, nu=1, mu=0, sigma=1) + self.log_precision_obj.sum()
        w_out = -(self.z_obj * self.z_obj * precision).sum() * 0.5
        out = w_out + precision_out
        return (out)

    def setup_parameter(self, obj, name, shape):
        self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_precision_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        setattr(obj, "z_obj", self.z_obj)
        setattr(obj, "log_precision_obj", self.log_precision_obj)
        return ()