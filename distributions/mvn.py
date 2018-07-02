from abstract.abstract_class_V import V
import torch,math
import torch.nn as nn
from torch.autograd import Variable
# precision_type = 'torch.DoubleTensor'
# #precision_type = 'torch.FloatTensor'
# torch.set_default_tensor_type(precision_type)


class V_mvn(V):
    def __init__(self,precision_type,input_data):
        self.input_data = input_data
        super(V_mvn, self).__init__(precision_type=precision_type)
    def V_setup(self):
        self.explicit_gradient = False
        self.need_higherorderderiv = False
        self.Sigma_inv = torch.from_numpy(self.input_data["input"]).type(self.precision_type)
        self.n = self.Sigma_inv.shape[0]
        self.beta = nn.Parameter(torch.zeros(self.n),requires_grad=True)

        self.Sigma_inv = Variable(self.Sigma_inv,requires_grad=False)
        return()
    def forward(self):
        # returns -log posterior
        out = 0.5 * torch.dot(self.beta,self.Sigma_inv.mv(self.beta))
        return(out)

    def load_explicit_gradient(self):
        out = self.Sigma_inv.mv(self.beta-self.mu)
        return(out)


