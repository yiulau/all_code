from distributions.neural_nets.priors.base_class import base_prior
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# horseshoe prior ncp parametrization for the model weight
# ncp parametrization for local lamb and global tau
class horseshoe_ard(base_prior):
    def __init__(self,obj,name,shape):
        self.setup_parameter(obj, name, shape)
        super(horseshoe_ard, self).__init__()

    def get_val(self):
        w_row_list = [None]*self.num_units
        for i in range(self.num_units):
            param_dict = self.param_list_by_units[i]
            local_r2 = torch.exp(param_dict["log_local_r2_obj"])
            global_r2 = torch.exp(param_dict["log_global_r2_obj"])
            tau = param_dict["global_r1_obj"] * torch.sqrt(global_r2)
            lamb = param_dict["local_r1_obj"] * torch.sqrt(local_r2)
            w_row = param_dict["z_obj"] * lamb * tau
            w_row_list[i] = w_row
        w_obj = torch.cat(w_row_list,0)
        return (w_obj)

    def get_out(self):
        out = 0
        for i in range(self.num_units):
            param_dict = self.param_list_by_units[i]
            z_out = -(param_dict["z_obj"] * param_dict["z_obj"]).sum() * 0.5
            local_r1_out = -(param_dict["local_r1_obj"] * param_dict["local_r1_obj"]).sum() * 0.5
            global_r1_out = -(param_dict["global_r1_obj"] * param_dict["global_r1_obj"]).sum() * 0.5
            local_r2 = torch.exp(param_dict["log_local_lamb_obj"])
            global_r2 = torch.exp(param_dict["log_global_tau_obj"])

            local_r2_out = log_inv_gamma_density(x=local_r2, alpha=0.5, beta=0.5) + param_dict["log_local_lamb_obj"].sum()
            global_r2_out = log_inv_gamma_density(x=global_r2, alpha=0.5,beta=0.5) + param_dict["log_global_tau_obj"].sum()
            out += z_out + local_r2_out + global_r2_out + local_r1_out + global_r1_out
        return (out)

    def setup_parameter(self, obj, name, shape):

        self.num_units = shape[0]
        self.in_units = shape[1]
        self.param_list_by_units = []

        for i in range(self.num_units):
            z_obj = nn.Parameter(torch.zeros(1,self.in_units), requires_grad=True)
            local_r1_obj = nn.Parameter(torch.zeros(1,self.in_units), requires_grad=True)
            log_local_r2_obj = nn.Parameter(torch.zeros(1,self.in_units), requires_grad=True)
            global_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
            log_global_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
            param_dict = {"z_obj":z_obj,"local_r1_obj":local_r1_obj,"log_local_r2_obj":log_local_r2_obj,"global_r1_obj":global_r1_obj,
                          "log_global_r2_obj":log_global_r2_obj}
            self.param_list_by_units.append(param_dict)

            setattr(obj, "z_obj"+"unit{}".format(i), self.z_obj)
            setattr(obj, "local_r1_obj"+"unit{}".format(i), self.local_r1_obj)
            setattr(obj, "log_local_r2_obj"+"unit{}".format(i), self.log_local_r2_obj)
            setattr(obj, "global_r1_obj"+"unit{}".format(i), self.global_r1_obj)
            setattr(obj, "log_global_r2_obj"+"unit{}".format(i), self.log_global_r2_obj)
        return ()