import torch,numpy, abc
import torch.nn as nn
# class base_prior(object):
#     __metaclass__ = abc.ABCMeta
#
#     @abc.abstractmethod
#     def setup_parameter(self):
#         return()
#
#     @abc.abstractmethod
#     def get_val(self):
#         return()
#
#     @abc.abstractmethod
#     def get_out(self):
#         return()
#
#     def __init__(self, obj, name, shape, prior_obj):
#         self.prior_obj = prior_obj
#         assert not hasattr(obj, name)
#         if prior_obj.model_param == "cp":
#             self.w_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#         else:
#             self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#
#         if prior_obj.name == "hs":
#             if prior_obj.prior_cp == "ncp":
#                 # self.z_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
#                 self.local_r1_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#                 self.log_local_r2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#                 self.global_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
#                 self.log_global_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
#
#             else:
#                 self.local_lamb_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#                 self.global_lamb_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
#
#         elif prior_obj.name == "rhorseshoe":
#             if prior_obj.prior_cp == "ncp":
#                 # self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#                 self.local_r1_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#                 self.log_local_r2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#                 self.global_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
#                 self.log_global_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
#                 self.c_z = nn.Parameter(torch.zeros(1), requires_grad=True)
#                 self.c_log_tau = nn.Parameter(torch.zeros(1), requires_grad=True)
#             else:
#                 self.local_lamb_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#                 self.global_lamb_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
#
#         else:
#             raise ValueError("unknown prior")
#
#     def get_val(self):
#         if self.prior_obj["name"] == "horseshoe_ncp":
#             lamb = torch.sqrt(torch.exp(self.log_local_r2_obj)) * self.local_r1_obj
#             tau = torch.sqrt(torch.exp(self.log_global_r2_obj)) * self.global_r1_obj
#             w = self.z_obj * lamb * tau
#
#         elif self.prior_obj["name"] == "rhorseshoe":
#             c = self.c_z * torch.exp(self.c_log_tau)
#             lamb = torch.sqrt(torch.exp(self.log_local_r2_obj)) * self.local_r1_obj
#             tau = torch.sqrt(torch.exp(self.log_global_r2_obj)) * self.global_r1_obj
#             lamb_tilde = c * c * lamb * lamb / (c * c + tau * tau * lamb * lamb)
#             w = self.z_obj * lamb_tilde * tau
#
#         return (w)
#
#     def get_out(self):
#         if self.prior_obj["name"] == "hs":
#             if self.prior_obj.hyper_param == "ncp":
#                 z_out = (self.z_obj * self.z_obj).sum()
#                 local_r1_out = (self.local_r1_obj * self.local_r1_obj).sum()
#                 global_r1_out = (self.global_r1_obj * self.global_r1_obj).sum()
#                 local_r2_out = log_inv_gamma_density(torch.exp(self.log_local_r2_obj), 0.5 + 1,
#                                                      0.5 + 1) + self.log_local_r2_obj
#                 local_r2_out = local_r2_out.sum()
#                 global_r2_out = log_inv_gamma_density(torch.exp(self.log_global_r2_obj), 0.5 + 1,
#                                                       0.5 + 1) + self.log_global_r2_obj
#                 global_r2_out = global_r2_out.sum()
#                 out = z_out + local_r1_out + global_r1_out + local_r2_out + global_r2_out
#
#         elif self.prior_obj["name"] == "rhorseshoe":
#             if self.prior_obj.hyper_param == "ncp":
#                 c_z_out = (self.c_z * self.c_z).sum() * 0.5
#                 c_tau = torch.exp(self.c_log_tau)
#                 c_tau_out = log_inv_gamma_density(c_tau, 1, 1) + self.c_log_tau
#                 z_out = (self.z_obj * self.z_obj).sum()
#                 local_r1_out = (self.local_r1_obj * self.local_r1_obj).sum()
#                 global_r1_out = (self.global_r1_obj * self.global_r1_obj).sum()
#                 local_r2_out = log_inv_gamma_density(torch.exp(self.log_local_r2_obj), 0.5 + 1,
#                                                      0.5 + 1) + self.log_local_r2_obj
#                 local_r2_out = local_r2_out.sum()
#                 global_r2_out = log_inv_gamma_density(torch.exp(self.log_global_r2_obj), 0.5 + 1,
#                                                       0.5 + 1) + self.log_global_r2_obj
#                 global_r2_out = global_r2_out.sum()
#                 out = z_out + local_r1_out + global_r1_out + local_r2_out + global_r2_out
#
#         return (out)
#

class base_prior_new(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def setup_parameter(self,obj,name,shape):
        return()

    @abc.abstractmethod
    def get_val(self):
        return()

    @abc.abstractmethod
    def get_out(self):
        return()

    @abc.abstractclassmethod
    def get_param(name_list):
        return()

    def init_dim(self):
        relevant_param_dim = 0
        len_list = []
        shape_list = []
        temp_list = self.get_param(name_list=self.relevant_param_tuple)
        #print(self.relevant_param_tuple)
        #print(temp_list)
        #exit()
        for i in range(len(self.relevant_param_tuple)):
            shape_list.append(temp_list[i].shape)
            this_dim = len(temp_list[i].view(-1))
            relevant_param_dim += this_dim
            len_list.append(this_dim)
        self.relevant_param_dim = relevant_param_dim
        self.len_list = len_list
        self.shape_list = shape_list
        #print(self.len_list)
        #print(self.relevant_param_dim)
        #print(self.shape_list)
        #exit()
        return()
    def get_all_param_flattened(self):
        if hasattr(self,"relevant_param_dim"):
            out = torch.zeros(self.relevant_param_dim)
        else:
           self.init_dim()
        out = torch.zeros(self.relevant_param_dim)

        cur = 0
        temp_list = self.get_param(name_list=self.relevant_param_tuple)
        for i in range(len(self.relevant_param_tuple)):
            out[cur:cur+self.len_list[i]] = temp_list[i].view(-1)
            cur += self.len_list[i]

        return(out)

    def get_index_specific(self,name):
        assert name in self.relevant_param_tuple
        if not hasattr(self,"len_list"):
            self.init_dim()
        index = self.relevant_param_tuple.index(name)
        cur = 0
        for i in range(index):
            cur += self.len_list[i]
        indices = list(range(cur,cur+self.len_list[index]))
        return(indices)

    def get_indices_dict(self):
        if not hasattr(self,"len_list"):
            self.init_dim()
        cur = 0
        out = {}
        for i in range(len(self.relevant_param_tuple)):
            indices = list(range(cur, cur + self.len_list[i]))
            out.update({self.relevant_param_tuple[i]:indices})
            cur += self.len_list[i]
        return(out)