from distributions.new_base_V_class import new_base_V_class
import torch.nn as nn
import torch, abc
class hs_model(new_base_V_class):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(hs_model, self).__init__()


    def V_setup(self,y,X,nu):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.dim = X.shape[1]
        self.log_sigma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.y = y
        self.X = X
        self.nu = nu
        self.prepare_prior()
        return()

    def log_likelihood(self):
        sigma = torch.exp(self.log_sigma)
        w = self.get_beta()[0]
        outy = -(self.y - (self.X.mv(w))) * (self.y - (self.X.mv(w))) / (sigma * sigma) * 0.5
        outhessian = - self.log_sigma
        out = outy + outhessian
        return(out)

    @abc.abstractmethod
    def log_prior(self):
        pass
    @abc.abstractmethod
    def prepare_prior(self):
        pass





