from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# one horseshoe prior ard for each hidden unit (weights entering the unit )in the layer
# one global scale parameter for each unit - num_units in total
# local scale for weights entering the unit -
# ncp parametrization for the model weight
# ncp parametrization for local lamb and global tau
# tau = N(0,global_scale^2)
class horseshoe_ard(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1):
        self.global_scale = global_scale
        self.nu = nu
        self.name = name
        self.relevant_param_tuple = ("w", "lamb", "tau")

        self.setup_parameter(obj, name, shape)
        super(horseshoe_ard, self).__init__()

    def get_val(self):
        w_row_list = [None]*self.num_units
        for i in range(self.num_units):
            param_dict = self.param_list_by_units[i]
            local_r2 = torch.exp(param_dict["log_local_r2_obj"])
            global_r2 = torch.exp(param_dict["log_global_r2_obj"])
            tau = param_dict["global_r1_obj"] * torch.sqrt(global_r2) * self.global_scale
            lamb = param_dict["local_r1_obj"] * torch.sqrt(local_r2)
            w_row = param_dict["z_obj"] * lamb * tau
            w_row_list[i] = w_row
        w_obj = torch.cat(w_row_list,0)
        return (w_obj)

    def get_val_alt(self):
        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.agg_log_global_r1_obj)
        global_r2 = torch.exp(self.agg_log_global_r2_obj)
        tau =  global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r1)
        w_obj = self.agg_z_obj * lamb * tau
        return(w_obj)

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

    def get_out_alt(self):
        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.agg_log_global_r1_obj)
        global_r2 = torch.exp(self.agg_log_global_r2_obj)
        z_out = -(self.agg_z_obj*self.agg_z_obj).sum()*0.5
        local_r1_out = -(local_r1*local_r1).sum() * 0.5
        global_r1_out = -(global_r1*global_r1).sum() * 0.5
        local_r2_out = log_inv_gamma_density(x=local_r2,alpha=0.5*self.nu,beta=0.5*self.nu) + self.agg_log_local_r2_obj.sum()
        global_r2_out = log_inv_gamma_density(x=global_r2,alpha=0.5*self.nu,beta=0.5*self.nu) + self.agg_log_global_r2_obj.sum()
        out = z_out + local_r1_out + local_r2_out + global_r1_out + global_r2_out
        return(out)


    def get_unit_scales(self):
        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.agg_log_global_r1_obj)
        global_r2 = torch.exp(self.agg_log_global_r2_obj)
        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r1)

        out  = torch.sqrt((tau * tau * lamb * lamb).sum(dim=1))
        return(out)
    def setup_parameter(self, obj, name, shape):

        self.num_units = shape[0]
        self.in_units = shape[1]
        self.param_list_by_units = []

        self.agg_z_obj = nn.Parameter(torch.zeros(self.num_units,self.in_units),requires_grad=True)
        self.agg_log_local_r1_obj = nn.Parameter(torch.zeros(self.num_units,self.in_units),requires_grad=True)
        self.agg_log_local_r2_obj = nn.Parameter(torch.zeros(self.num_units,self.in_units),requires_grad=True)
        self.agg_log_global_r1_obj = nn.Parameter(torch.zeros(self.num_units,1),requires_grad=True)
        self.agg_log_global_r2_obj = nn.Parameter(torch.zeros(self.num_units,1),requires_grad=True)


        for i in range(self.num_units):
            z_obj = self.agg_z_obj[i,:]
            log_local_r1_obj = self.agg_log_local_r1_obj[i,:]
            log_local_r2_obj = self.agg_log_local_r2_obj[i,:]
            log_global_r1_obj = self.agg_log_global_r1_obj[i,0]
            log_global_r2_obj = self.agg_log_global_r2_obj[i,0]
            param_dict = {"z_obj":z_obj,"log_local_r1_obj":log_local_r1_obj,"log_local_r2_obj":log_local_r2_obj,
                          "global_r1_obj":log_global_r1_obj,"log_global_r2_obj":log_global_r2_obj}
            self.param_list_by_units.append(param_dict)

        setattr(obj, name+"_agg_z_obj", self.agg_z_obj)
        setattr(obj, name+"_agg_log_local_r1_obj", self.agg_log_local_r1_obj)
        setattr(obj, name+"_agg_log_local_r2_obj", self.agg_log_local_r2_obj)
        setattr(obj, name+"_agg_log_global_r1_obj", self.agg_log_global_r1_obj)
        setattr(obj, name+"_agg_log_global_r2_obj", self.agg_log_global_r2_obj)
        return ()

    def get_param(self, name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple
        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.agg_log_global_r1_obj)
        global_r2 = torch.exp(self.agg_log_global_r2_obj)
        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r1)
        w_obj = self.agg_z_obj * lamb * tau

        out_list = [None] * len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = w_obj
            elif name == "tau":
                out = tau
            elif name == "lamb":
                out = lamb
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return (out_list)