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


