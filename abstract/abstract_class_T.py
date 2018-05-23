import torch,abc,numpy
from torch.autograd import Variable
from abstract.abstract_class_point import point
from general_util.pytorch_util import general_load_point,general_assert_syncro
from general_util.pytorch_util import general_load_flattened_tensor_to_param,general_load_param_to_flattened
class T(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,linkedV):
        #self.metric = metric
        self.linkedV = linkedV
        self.dim = self.linkedV.dim
        self.store_lens = self.linkedV.store_lens
        self.store_shapes = self.linkedV.store_shapes
        self.num_var = self.linkedV.num_var
        self.list_tensor = numpy.empty(self.linkedV.num_var,dtype=type(self.linkedV.list_tensor[0]))
        for i in range(len(self.list_tensor)):
            self.list_tensor[i] = torch.zeros(self.linkedV.list_tensor[i].shape)
        self.list_var = numpy.empty(self.linkedV.num_var, dtype=type(self.linkedV.list_var[0]))
        for i in range(len(self.list_var)):
            self.list_var[i] = Variable(self.list_tensor[i],requires_grad=False)
        # need gradient for every metric except unit_e
        #self.gradient = numpy.empty(len(self.store_shapes), dtype=torch.FloatTensor)
        # definitely need faltten for softabs,
        # for other metrics depend on
        self.need_flatten = self.linkedV.need_flatten
        if self.need_flatten:
            #self.flattened_gradient = torch.zeros(self.dim)
            self.flattened_tensor = torch.zeros(self.dim)
        else:
            self.flattened_tensor = self.list_tensor[0]
            if self.metric.name=="unit_e":
                self.flattened_gradient = self.flattened_tensor
        #self.p_point = point(T=self)
        #return()
    # def load_listp_to_flattened(self,list_tensor,target_tensor):
    #     cur = 0
    #     for i in range(len(self.p)):
    #         target_tensor[cur:(cur + self.store_lens[i])] = self.list_tensor[i].view(self.store_lens[i])
    #         cur = cur + self.store_lens[i]
    # def load_flattened_tenosr_to_target_list(self,target_list,flattened_tensor):
    #     cur = 0
    #     for i in range(len(target_list)):
    #         target_list[i].copy_(flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i]))
    #         cur = cur + self.store_lens[i]
    #     return ()

    # def load_point(self,p_point):
    #     for i in range(self.num_var):
    #         # convert to copy_ later
    #         self.list_var[i].data.copy_(p_point.list_tensor[i])
    #     self.flattened_tensor.copy_(p_point.flattened_tensor)
    #     return()
    def load_flattened_tensor_to_param(self,flattened_tensor=None):
        general_load_flattened_tensor_to_param(obj=self,flattened_tensor=flattened_tensor)
        return()
    def load_param_to_flattened(self):
        general_load_param_to_flattened(obj=self)
        return()
    def load_point(self,p_point):
        general_load_point(obj=self,point_obj=p_point)
        return()

    def assert_syncro(self):
        return (general_assert_syncro(self))
    @abc.abstractmethod
    def evaluate_scalar(self,q_point=None,p_point=None):
        # returns T(q,p) -- scalar - float or double
        return()

    @abc.abstractmethod
    def dp(self):
        # takes and returns flattened_tensor
        return()
    @abc.abstractmethod
    def dtaudp(self,lam=None,Q=None):
        return()

    @abc.abstractmethod
    def dtaudq(self,alpha=None,H_=None,dH=None):
        # flattened_p,q
        return()

    @abc.abstractmethod
    def generate_momentum(self,lam=None,Q=None):
        # returns generated in a list that has the same shape as the original variables
        return()


