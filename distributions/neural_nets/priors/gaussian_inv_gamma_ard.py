from distributions.neural_nets.priors.base_class import base_prior_new
import torch.nn as nn
import torch
from general_util.pytorch_random import log_student_t_density, log_inv_gamma_density


# horseshoe prior ard for entire layer
# one global scale parameter for the entire layer
# local scale shared for each unit
# ncp parametrization for the model weight
# ncp parametrization for local lamb and global tau
class gaussian_inv_gamma_ard(base_prior_new):
    def __init__(self,obj,name,shape,global_scale=1,global_df=1):
        self.global_df = global_df
        self.global_scale = global_scale
        self.name = name
        self.relevant_param_tuple = ("w", "sigma2")

        self.setup_parameter(obj, name, shape)
        super(gaussian_inv_gamma_ard, self).__init__()

    def get_val(self):
        w_row_list = [None]*self.num_units
        for i in range(self.num_units):
            param_dict = self.param_list_by_units[i]
            sigma2 = torch.exp(param_dict["log_sigma2_obj"])
            w_row = param_dict["z_obj"] * torch.sqrt(sigma2)
            w_row_list[i] = w_row
        w_obj = torch.cat(w_row_list,0)
        return (w_obj)

    def get_val_alt(self):
        sigma2 = torch.exp(self.agg_log_sigma2_obj)
        w_obj = self.agg_z_obj * torch.sqrt(sigma2)

        return(w_obj)
    def get_out(self):
        out = 0
        for i in range(self.num_units):
            param_dict = self.param_list_by_units[i]
            z_out = -(param_dict["z_obj"] * param_dict["z_obj"]).sum() * 0.5
            sigma2 = torch.exp(param_dict["log_sigma2_obj"])
            sigma2_out = log_inv_gamma_density(x=sigma2, alpha=0.5*self.global_df, beta=0.5*self.global_scale) + param_dict["log_sigma2_obj"].sum()
            out += z_out + sigma2_out
        return (out)
    def get_out_alt(self):

        z_out = -(self.agg_z_obj * self.agg_z_obj).sum() * 0.5
        sigma2 = torch.exp(self.agg_log_sigma2_obj)
        sigma2_out = log_inv_gamma_density(x=sigma2, alpha=0.5*self.global_df, beta=0.5*self.global_scale) + self.agg_log_sigma2_obj.sum()
        out = z_out + sigma2_out
        return(out)


    def get_unit_scales(self):
        # return sum for weights entering each unit
        agg_sigma2 = torch.exp(self.agg_log_sigma2_obj)
        out = torch.sqrt(agg_sigma2* self.in_units)
        return(out)

    def setup_parameter(self, obj, name, shape):

        self.num_units = shape[0]
        self.in_units = shape[1]
        self.param_list_by_units = []

        self.agg_z_obj = nn.Parameter(torch.zeros(self.num_units,self.in_units),requires_grad=True)
        self.agg_log_sigma2_obj = nn.Parameter(torch.zeros(self.num_units,1),requires_grad=True)
        for i in range(self.num_units):
            z_obj = self.agg_z_obj[i:i+1,:]
            log_sigma2_obj = self.agg_log_sigma2_obj[i:i+1,0]
            param_dict = {"z_obj":z_obj,"log_sigma2_obj":log_sigma2_obj}
            self.param_list_by_units.append(param_dict)

        setattr(obj, name+"_agg_z_obj", self.agg_z_obj)
        setattr(obj, name+"_log_sigma2_obj", self.agg_log_sigma2_obj)
        return ()

    def get_param(self,name_list):
        for name in name_list:
            assert name in self.relevant_param_tuple
        sigma2 = torch.exp(self.agg_log_sigma2_obj)
        w_obj = self.agg_z_obj * torch.sqrt(sigma2)
        out_list = [None]*len(name_list)
        for i in range(len(name_list)):
            name = name_list[i]
            if name == "w":
                out = w_obj
            elif name =="sigma2":
                out = sigma2
            else:
                raise ValueError("unknown name")
            out_list[i] = out.data.clone()


        return(out_list)