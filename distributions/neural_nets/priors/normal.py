from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch,math
from general_util.pytorch_random import log_student_t_density

class normal(base_prior_new):
    # fixed non-unity variance
    def __init__(self,obj,name,shape,var):
        self.var = var
        self.sd = math.sqrt(self.var)
        self.setup_parameter(obj,name,shape)
        #super(horseshoe_1, self).__init__()

    def get_val(self):
        w_obj = self.z_obj * self.sd
        return(w_obj)

    def get_out(self):
        z_out = -(self.z_obj*self.z_obj).sum()*0.5
        out = z_out
        return(out)

    def setup_parameter(self,obj, name, shape):
        self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        setattr(obj,"z_obj_"+name,self.z_obj)
        return()

