from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density
# horseshoe prior ncp parametrization for the model weight
# cp parametrization for local lamb and global tau
class standard_normal(base_prior_new):
    def __init__(self,obj,name,shape):

        self.relevant_param_tuple = ("w")
        self.name = name
        self.setup_parameter(obj,name,shape)
        #super(horseshoe_1, self).__init__()


    def get_val(self):
        return(self.w_obj)

    def get_out(self):
        w_out = -(self.w_obj*self.w_obj).sum()*0.5
        out = w_out
        return(out)

    def setup_parameter(self,obj, name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        setattr(obj,name+"_w_obj",self.w_obj)

        return()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple

        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = self.w_obj
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)


#horseshoe_1(obj={},name="t",shape=[15])