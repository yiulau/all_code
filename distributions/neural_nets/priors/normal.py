from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch,math
from general_util.pytorch_random import log_student_t_density
# ncp version

class normal(base_prior_new):
    # fixed non-unity variance
    def __init__(self,obj,name,shape,global_scale=1):
        self.var = global_scale*global_scale
        self.sd = math.sqrt(self.var)
        self.name = name
        self.relevant_param_tuple = ("w")
        self.setup_parameter(obj,name,shape)
        super(normal, self).__init__()

    def get_val(self):
        w_obj = self.z_obj * self.sd
        return(w_obj)

    def get_out(self):
        z_out = -(self.z_obj*self.z_obj).sum()*0.5
        out = z_out
        return(out)

    def setup_parameter(self,obj, name, shape):
        self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        setattr(obj,name+"_z_obj",self.z_obj)
        return()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple

        w_obj = self.z_obj * self.sd
        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = w_obj
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)

