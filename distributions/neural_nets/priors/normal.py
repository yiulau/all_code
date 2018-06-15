from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density

class normal(base_prior_new):
    # fixed non-unity variance
    def __init__(self,obj,name,shape,var):
        self.var = var
        self.setup_parameter(obj,name,shape)
        #super(horseshoe_1, self).__init__()

    def get_val(self):
        return(self.w_obj)

    def get_out(self):
        w_out = -(self.w_obj*self.w_obj/self.var).sum()*0.5
        out = w_out
        return(out)

    def setup_parameter(self,obj, name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        setattr(obj,"w_obj",self.z_obj)
        return()

