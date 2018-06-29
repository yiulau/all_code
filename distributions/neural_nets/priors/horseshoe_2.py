from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density
# horseshoe prior cp parametrization for the model weight
# cp parametrization for local lamb and global tau
class horseshoe_2(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1):
        self.global_scale = global_scale
        self.nu = nu
        self.name = name
        self.relevant_param_tuple = ("w", "lamb", "tau")
        self.setup_parameter(obj,name,shape)
        #super(horseshoe_2, self).__init__()

    def get_val(self):
        return(self.w_obj)

    def get_out(self):

        lamb2 = torch.exp(self.log_lamb2_obj)
        tau2 = torch.exp(self.log_tau2_obj)
        lamb2_out = log_student_t_density(x=lamb2,nu=1,mu=0,sigma=1) + self.log_lamb2_obj.sum()
        tau2_out = log_student_t_density(x=tau2,nu=self.nu,mu=0,sigma=self.global_scale) + self.log_tau2_obj.sum()
        w_out = -(self.w_obj*self.w_obj/(lamb2*tau2)).sum()*0.5 -(torch.log(lamb2*tau2)).sum()*0.5

        out = w_out + lamb2_out + tau2_out
        return(out)

    def setup_parameter(self,obj,name, shape):
        self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_lamb2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.log_tau2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)

        setattr(obj,name+"_w_obj",self.w_obj)
        setattr(obj,name+"_lamb2_obj",self.log_lamb2_obj)
        setattr(obj,name+"_tau2_obj",self.log_tau2_obj)
        return()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple

        lamb = torch.sqrt(torch.exp(self.log_lamb2_obj))
        tau = torch.sqrt(torch.exp(self.log_tau2_obj))

        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = self.w_obj
            elif name =="tau":
                out = tau
            elif name == "lamb":
                out = lamb
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)