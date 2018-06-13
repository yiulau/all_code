from abstract.abstract_class_V import V
import torch,math
import torch.nn as nn
from torch.autograd import Variable
# precision_type = 'torch.DoubleTensor'
# #precision_type = 'torch.FloatTensor'
# torch.set_default_tensor_type(precision_type)


class V_2dnormal(V):
    def __init__(self,precision_type="torch.DoubleTensor"):
        super(V, self).__init__(precision_type=precision_type)
    def V_setup(self):
        self.n = 2
        self.explicit_gradient = False
        self.need_higherorderderiv = False
        self.beta = nn.Parameter(torch.zeros(self.n),requires_grad=True)
        self.Sigma = torch.zeros(self.n,self.n)
        self.Sigma[0,0] = 1.5
        self.Sigma[0,1] = 0.5
        self.Sigma[1,0] = self.Sigma[0,1]
        self.Sigma[1,1] = 2.5
        self.mu = torch.zeros(self.n)
        self.mu[0] = 1.5
        self.mu[1] = -2.3
        self.mu = Variable(self.mu,requires_grad=False)
        self.Sigma_inv = Variable(torch.inverse(self.Sigma),requires_grad=False)
        #self.n = n
        # beta[n-1] = y ,
        # beta[:(n-1)] = x
        return()

    def forward(self):
        # returns -log posterior
        out = 0.5 * torch.dot(self.beta-self.mu,self.Sigma_inv.mv(self.beta-self.mu))
        return(out)

    def load_explicit_gradient(self):
        out = self.Sigma_inv.mv(self.beta-self.mu)
        return(out)


