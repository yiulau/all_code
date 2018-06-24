# base class for any model that has data and parameters to be estimated in bayseian inference
from abstract.abstract_class_V import V
import abc,torch,math
class bayes_model_class(V):
    __metaclass__ = abc.ABCMeta

    def __init__(self,input_data,precision_type):
        self.input_data = input_data
        super(bayes_model_class, self).__init__(precision_type=precision_type)

    def p_y_given_theta(self, observed_point, posterior_point):

        log_p_y_given_theta = self.log_p_y_given_theta(observed_point,posterior_point)
        out = math.exp(log_p_y_given_theta)
        #out = torch.exp(-out)
        return (out)

    @abc.abstractmethod
    def log_p_y_given_theta(self, observed_point, posterior_point):
        # self.load_point(posterior_point)
        # #out = -self.forward(input=observed_point)
        # out = torch.log(self.p_y_given_theta(input=observed_point))
        return()

    def get_all_param_flattened(self):
        if hasattr(self,"relevant_param_dim"):
            out = torch.zeros(self.relevant_param_dim)
        else:
            relevant_param_dim = 0
            len_list = []
            shape_list = []
            temp_list = self.get_param(name_list=self.relevant_param_tuple)
            for i in range(len(self.relevant_param_tuple)):
                shape_list.append(temp_list[i].shape)
                this_dim = len(temp_list[i].view(-1))
                relevant_param_dim += this_dim
                len_list.append(this_dim)
            self.relevant_param_dim = relevant_param_dim
            self.dim_list = len_list
            self.shape_list = shape_list
            out = torch.zeros(self.relevant_param_dim)

        cur = 0
        temp_list = self.get_param(name_list=self.relevant_param_tuple)
        for i in range(len(self.relevant_param_tuple)):
            out[cur:cur+self.dim_list[i]] = temp_list[i].view(-1)
            cur += self.dim_list[i]

        return(out)

    def get_indices(self,name):
        assert name in self.relevant_param_tuple
        index = self.relevant_param_tuple.index(name)
        cur = 0
        for i in range(index):
            cur += self.dim_list[i]
        indices = list(range(cur,cur+self.dim_list[index]))
        return(indices)
