import torch
import torch.nn as nn
from abstract.abstract_class_V import V
from torch.autograd import Variable
from input_data.convert_data_to_dict import get_data_dict
from explicit.general_util import logsumexp_torch

# precision_type = 'torch.DoubleTensor'
# #precision_type = 'torch.FloatTensor'
# torch.set_default_tensor_type(precision_type)


class V_linear_regression(V):
    def __init__(self,precision_type="torch.DoubleTensor"):

        super(V_linear_regression, self).__init__(precision_type=precision_type)
    def V_setup(self):
        input_npdata = get_data_dict("boston")
        self.y_np = input_npdata["target"]
        self.X_np = input_npdata["input"]
        self.dim = self.X_np.shape[1]
        self.num_ob = self.X_np.shape[0]
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.zeros(self.dim),requires_grad=True)
        self.y = Variable(torch.from_numpy(self.y_np),requires_grad=False).type(self.precision_type)
        self.X = Variable(torch.from_numpy(self.X_np),requires_grad=False).type(self.precision_type)
        # include
        self.sigma =1

        return()

    def forward(self):
        print(len(self.beta))

        likelihood = -(((self.y - self.X.mv(self.beta))*(self.y-self.X.mv(self.beta)))).sum()*0.5
        prior = -torch.dot(self.beta, self.beta)/(self.sigma*self.sigma) * 0.5
        print("likelihood {}".format(likelihood))
        print("prior {}".format(prior))
        print("beta {}".format(self.beta))
        posterior = prior + likelihood
        out = -posterior
        return(out)

    def predict(self,test_samples):
        X = torch.from_numpy(test_samples)
        out = torch.zeros(X.shape[0])
        out = torch.mv(X, self.beta.data)
        return(out)
    def load_explicit_gradient(self):
        out = torch.zeros(self.dim)
        X = self.X.data
        beta = self.beta.data
        y = self.y.data
        pihat = torch.sigmoid(torch.mv(X, beta))
        out = -X.t().mv(y - pihat) + beta
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.dim,self.dim)
        X = self.X.data
        beta = self.beta.data
        y = self.y.data
        pihat = torch.sigmoid(torch.mv(X, beta))
        out = torch.mm(X.t(), torch.mm(X.t(), torch.diag(pihat * (1. - pihat))).t()) + torch.diag(
            torch.ones(len(beta)))

        return(out)


    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.dim,self.dim,self.dim)
        X = self.X.data
        beta = self.beta.data
        y = self.y.data
        pihat = torch.sigmoid(torch.mv(X, beta))
        for i in range(self.dim):
            out[i, :, :] = (
            torch.mm(X.t(), torch.diag(pihat * (1. - pihat))).mm(torch.diag(1. - 2 * pihat) * X[:, i]).mm(X))

        return(out)

    def load_explicit_diagH(self):
        out = self.load_explicit_H()
        return (torch.diag(out))
    def load_explicit_graddiagH(self):
        temp = self.load_explicit_dH()
        out = torch.zeros(self.dim,self.dim)
        for i in range(self.dim):
            out[i,:] = torch.diag(temp[i,:,:])
        #out = out.t()
        return(out)