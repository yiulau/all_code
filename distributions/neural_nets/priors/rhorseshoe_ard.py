from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# regularized horseshoe prior ard for entire layer
# one global scale parameter for the entire layer
# local scale shared for each unit
# ncp parametrization for the model weight
# ncp parametrization for local lamb and global tau
# tau = C+(0,global_scale^2) , C+ is half-cauchy

class rhorseshoe_ard(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,nu=1,slab_df=4,slab_scale=2):
        self.global_scale = global_scale
        self.nu = nu
        self.slab_df = slab_df
        self.slab_scale = slab_scale
        self.name = name
        self.relevant_param_tuple = ("w","lamb","lamb_tilde","c","tau")
        self.setup_parameter(obj,name, shape)
        super(rhorseshoe_ard, self).__init__()

    def get_val_alt(self):
        w_row_list = [None]*self.num_units
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        for i in range(self.num_units):
            param_dict = self.param_list_by_units[i]
            local_r1 = torch.exp(param_dict["log_local_r1_obj"])
            global_r1 = torch.exp(param_dict["log_global_r1_obj"])
            local_r2 = torch.exp(param_dict["log_local_r2_obj"])
            global_r2 = torch.exp(param_dict["log_global_r2_obj"])
            tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
            lamb = local_r1 * torch.sqrt(local_r2)
            lamb_tilde = torch.sqrt(c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb))
            w_row = param_dict["z_obj"] * lamb_tilde * tau
            w_row_list[i] = w_row
        w_obj = torch.cat(w_row_list,0)
        return (w_obj)

    def get_val(self):
        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)

        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = torch.sqrt(c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb))
        w_obj = self.agg_z_obj * lamb_tilde *tau

        return(w_obj)
    def get_out_alt(self):
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c_r1_out = -(c_r1 * c_r1).sum() * 0.5 + self.log_c_r1_obj
        c_alpha = self.slab_df / 2
        c_beta = self.slab_df * self.slab_scale * self.slab_scale / 2
        c_r2_out = log_inv_gamma_density(x=c_r2, alpha=c_alpha, beta=c_beta) + self.log_c_r2_obj.sum()
        out = c_r1_out + c_r2_out
        for i in range(self.num_units):
            param_dict = self.param_list_by_units[i]
            local_r1 = torch.exp(param_dict["log_local_r1_obj"])
            global_r1 = torch.exp(param_dict["log_global_r1_obj"])
            local_r2 = torch.exp(param_dict["log_local_r2_obj"])
            global_r2 = torch.exp(param_dict["log_global_r2_obj"])
            z_out = -(param_dict["z_obj"] * param_dict["z_obj"]).sum() * 0.5
            local_r1_out = -(local_r1 * local_r1).sum() * 0.5
            global_r1_out = -(global_r1 * global_r1).sum() * 0.5


            local_r2_out = log_inv_gamma_density(x=local_r2, alpha=0.5, beta=0.5) + param_dict["log_local_r2_obj"].sum()
            global_r2_out = log_inv_gamma_density(x=global_r2, alpha=0.5*self.nu,beta=0.5*self.nu) + param_dict["log_global_r2_obj"].sum()
            out += z_out + local_r2_out + global_r2_out + local_r1_out + global_r1_out
        return (out)
    def get_out(self):

        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        z_out = -(self.agg_z_obj*self.agg_z_obj).sum() * 0.5
        local_r1_out = -(local_r1*local_r1).sum() * 0.5 + self.agg_log_local_r1_obj.sum()
        global_r1_out = -(global_r1*global_r1).sum()*0.5 + self.log_global_r1_obj.sum()

        local_r2_out = log_inv_gamma_density(x=local_r2,alpha=0.5,beta=0.5) + self.agg_log_local_r1_obj.sum()
        global_r2_out = log_inv_gamma_density(x=global_r2,alpha=0.5*self.nu,beta=0.5*self.nu) + self.agg_log_local_r2_obj.sum()
        c_r1_out = -(c_r1 * c_r1).sum() * 0.5 + self.log_c_r1_obj
        c_alpha = self.slab_df / 2
        c_beta = self.slab_df * self.slab_scale * self.slab_scale / 2
        c_r2_out = log_inv_gamma_density(x=c_r2, alpha=c_alpha, beta=c_beta) + self.log_c_r2_obj.sum()
        out = z_out + local_r1_out + local_r2_out + global_r1_out + global_r2_out + c_r1_out + c_r2_out
        return(out)


    def get_unit_scales(self):
        # return sum tau lamb for weights entering each unit
        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)

        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = torch.sqrt(c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb))

        out = torch.sqrt(tau * tau * lamb_tilde * lamb_tilde * self.in_units)
        return(out)

    def setup_parameter(self, obj,name, shape):

        self.num_units = shape[0]
        self.in_units = shape[1]
        self.param_list_by_units = []

        self.agg_z_obj = nn.Parameter(torch.zeros(self.num_units,self.in_units),requires_grad=True)
        self.agg_log_local_r1_obj = nn.Parameter(torch.zeros(self.num_units,1),requires_grad=True)
        self.agg_log_local_r2_obj = nn.Parameter(torch.zeros(self.num_units,1),requires_grad=True)
        self.log_global_r1_obj = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.log_global_r2_obj = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.log_c_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_c_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
        for i in range(self.num_units):
            z_obj = self.agg_z_obj[i,:]
            log_local_r1_obj = self.agg_log_local_r1_obj[i:i+1,:]
            log_local_r2_obj = self.agg_log_local_r2_obj[i:i+1,:]
            log_global_r1_obj = self.log_global_r1_obj
            log_global_r2_obj = self.log_global_r2_obj
            param_dict = {"z_obj":z_obj,"log_local_r1_obj":log_local_r1_obj,"log_local_r2_obj":log_local_r2_obj,
                          "log_global_r1_obj":log_global_r1_obj,"log_global_r2_obj":log_global_r2_obj}
            self.param_list_by_units.append(param_dict)

        setattr(obj, name+"_agg_z_obj", self.agg_z_obj)
        setattr(obj, name+"_agg_log_local_r1_obj", self.agg_log_local_r1_obj)
        setattr(obj, name+"_agg_log_local_r2_obj", self.agg_log_local_r2_obj)
        setattr(obj, name+"_log_global_r1_obj", self.log_global_r1_obj)
        setattr(obj, name+"_log_global_r2_obj", self.log_global_r2_obj)
        setattr(obj, name+"_c_log_r1_obj", self.log_c_r1_obj)
        setattr(obj, name+"_c_log_r2_obj", self.log_c_r2_obj)
        return ()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple

        local_r1 = torch.exp(self.agg_log_local_r1_obj)
        local_r2 = torch.exp(self.agg_log_local_r2_obj)
        global_r1 = torch.exp(self.log_global_r1_obj)
        global_r2 = torch.exp(self.log_global_r2_obj)

        tau = global_r1 * torch.sqrt(global_r2) * self.global_scale
        lamb = local_r1 * torch.sqrt(local_r2)
        c_r1 = torch.exp(self.log_c_r1_obj)
        c_r2 = torch.exp(self.log_c_r2_obj)
        c = c_r1 * torch.sqrt(c_r2)
        lamb_tilde = torch.sqrt(c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb))
        w_obj = self.agg_z_obj * lamb_tilde * tau

        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = w_obj
            elif name =="tau":
                out = tau
            elif name == "lamb":
                out = lamb
            elif name=="lamb_tilde":
                out = lamb_tilde
            elif name=="c":
                out = c
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()
        return(out_list)