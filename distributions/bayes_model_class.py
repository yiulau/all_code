# base class for any model that has data and beta
from abstract.abstract_class_V import V
import abc,torch
class bayes_model_class(V):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(bayes_model_class, self).__init__()

    def p_y_given_theta(self, observed_point, posterior_point):
        self.load_point(posterior_point)
        out = self.forward(input=observed_point)
        out = torch.exp(-out)
        return (out.data[0])

    def log_p_y_given_theta(self, observed_point, posterior_point):
        self.load_point(posterior_point)
        out = -self.forward(input=observed_point)
        return (out.data[0])