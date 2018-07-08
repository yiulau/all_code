import numpy
import torch
import torch.nn as nn
from abstract.abstract_class_point import point
from torch.autograd import Variable, grad
from general_util.pytorch_util import get_list_stats, general_load_point,general_assert_syncro
from general_util.pytorch_util import general_load_param_to_flattened,general_load_flattened_tensor_to_param
from general_util.pytorch_util import isnan

# if need to define explicit gradient do it

class V(nn.Module):
    #def __init__(self,explicit_gradient):
    def __init__(self,precision_type):
        super(V, self).__init__()
        self.precision_type=precision_type
        torch.set_default_tensor_type(self.precision_type)
        self.V_setup()
        #print(self.precision_type)
        #print(self.X)
        #exit()
        if self.explicit_gradient is None:
            raise ValueError("self.explicit_gradient need to be defined in V_setup")

        #################################################################################

        self.decides_if_flattened()
        self.V_higherorder_setup()
        #self.q_point = point(V=self)
        self.diagnostics = None
    #@abc.abstractmethod
    #def V_setup(self):
        ### initialize function parameters
    #    return()
    #@abc.abstractmethod
    #def forward(self):
    #    return()
    #@abc.abstractmethod
    #def load_explicit_gradient(self):
    #    raise NotImplementedError()
    #    return()
    def get_model_dim(self):
        out = len(self.flattened_tensor)
        return(out)
    def evaluate_scalar(self,q_point=None):
        # return float or double
        if not q_point is None:
            self.load_point(q_point)
        else:
            pass
        return(self.forward().data[0])
    # def gradient(self):
    #     # load gradients to list(self.parameters()).grad
    #     if not self.explicit_gradient:
    #         o = self.forward()
    #         o.backward()
    #     else:
    #         self.load_explicit_gradient()

    def dq(self,q_flattened_tensor,input_data=None):
        self.load_flattened_tensor_to_param(q_flattened_tensor)
        if input_data is None:
            g = grad(self.forward(),self.list_var)
        else:
            g = grad(self.forward(input=input_data), self.list_var)
        # check for exploding gradient
        explode_grad = False
        for i in range(len(g)):
            #print(g)
            #exit()
            explode_grad = explode_grad or isnan(g[i].data)
            if explode_grad:
                return(None,True)
        out = torch.zeros(len(q_flattened_tensor))

        cur = 0
        for i in range(self.num_var):
            out[cur:(cur + self.store_lens[i])] = g[i].data.view(self.store_lens[i])
            cur = cur + self.store_lens[i]
        return(out,explode_grad)



    def getdV(self,q=None):
        # return list of pytorch variables containing the gradient
        if not q is None:
            self.load_point(q)
        if not self.need_flatten:
            g = grad(self.forward(),self.list_var,create_graph=True)[0]
        else:
            g = grad(self.forward(), self.list_var, create_graph=True)
        self.load_gradient(g)
        if not self.diagnostics is None:
            self.diagnostics.add_num_grad(1)
        return(g)

    def getdV_tensor(self,q=None):
        # return list of pytorch variables containing the gradient
        #if not q.V is self:
        #    raise ValueError("not the same V function object")
        if not q is None:
            self.load_point(q)
        if self.explicit_gradient:
            self.gradient_tensor.copy_(self.load_explicit_gradient())
        else:
            g = grad(self.forward(), self.list_var, create_graph=True)
            self.load_gradient(g)
            if not self.diagnostics is None:
                self.diagnostics.add_num_grad(1)
        return(self.gradient_tensor)


    def load_gradient(self,list_g):
        #print("abstract dV {}".format(list_g.data))
        if not self.need_flatten:
            self.gradient_tensor.copy_(list_g[0].data)
        else:
            cur = 0
            for i in range(self.num_var):
                self.gradient_tensor[cur:(cur + self.store_lens[i])] = list_g[i].data.view(self.store_lens[i])
                cur = cur + self.store_lens[i]
        return()

    def V_higherorder_setup(self):
        self.gradient_tensor = torch.zeros(self.dim)
        self.list_var = list(self.parameters())
        return()


    def decides_if_flattened(self):
        self.need_flatten = False
        self.list_var = list(self.parameters())
        self.num_var = len(self.list_var)
        self.list_tensor = numpy.empty(self.num_var,dtype=type(self.list_var[0].data))
        for i in range(len(self.list_tensor)):
            self.list_tensor[i] = self.list_var[i].data
        if self.num_var>1:
            self.need_flatten = True
        elif self.num_var==1:
            if len(list(self.list_tensor[0].shape))>1:
                self.need_flatten = True
        else:
            raise ValueError("count is >=2 but not 1")
        self.store_shapes,self.store_lens,self.dim,self.store_slices=get_list_stats(list_tensor=self.list_tensor)
        if self.need_flatten:
            self.flattened_tensor = torch.zeros(self.dim)
            cur = 0
            for i in range(self.num_var):
                self.flattened_tensor[cur:(cur + self.store_lens[i])] = self.list_tensor[i].view(self.store_lens[i])
                cur = cur + self.store_lens[i]
        else:
            self.flattened_tensor = self.list_var[0].data
        return()

    def load_flattened_tensor_to_param(self,flattened_tensor=None):
        general_load_flattened_tensor_to_param(obj=self,flattened_tensor=flattened_tensor)
        return()
    def load_param_to_flattened(self):
        general_load_param_to_flattened(obj=self)
        return()
    def load_point(self,q_point):
        general_load_point(obj=self,point_obj=q_point)
        return()
    def asssert_syncro(self):
        return(general_assert_syncro(self))


    def prepare_prior(self,prior_dict):
        if prior_dict["has_hypar"]:
            prior_dict["create_hypar_fun"](self)
        return()

    def get_name_dim(self,name):
        # returns number of dimensions = num of weight parameters + num of hyperparameters
        prior_obj = self.dict_parameters[name]

        return()





