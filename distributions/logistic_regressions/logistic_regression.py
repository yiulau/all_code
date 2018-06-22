import torch
import torch.nn as nn
from abstract.abstract_class_V import V
from torch.autograd import Variable
from distributions.bayes_model_class import bayes_model_class
from explicit.general_util import logsumexp_torch

# precision_type = 'torch.DoubleTensor'
# #precision_type = 'torch.FloatTensor'
# torch.set_default_tensor_type(precision_type)

class V_logistic_regression(bayes_model_class):
#class V_logistic_regression(V):
    def __init__(self,input_data,precision_type):
        #print(precision_type)
        #print(self.precision_type)
        #exit()
        super(V_logistic_regression, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.input_data["input"].shape[1]
        self.num_ob = self.input_data["input"].shape[0]
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.zeros(self.dim),requires_grad=True)
        self.y = Variable(torch.from_numpy(self.input_data["target"]),requires_grad=False).type(self.precision_type)
        self.X = Variable(torch.from_numpy(self.input_data["input"]),requires_grad=False).type(self.precision_type)
        # include
        self.sigma =1

        return()

    def forward(self):
        # if input is None:
        #     X = self.X
        #     y = self.y
        #
        # else:
        #     X = Variable(input["input"],requires_grad=False).type(self.precision_type)
        #     y = Variable(input["target"],requires_grad=False).type(self.precision_type)
        # num_ob = X.shape[0]
        # print(self.precision_type)
        # print(self.beta)
        # #print(X.data)
        # exit()
        likelihood = torch.dot(self.beta, torch.mv(torch.t(self.X), self.y)) - \
                     torch.sum(logsumexp_torch(Variable(torch.zeros(self.num_ob)), torch.mv(self.X, self.beta)))
        prior = -torch.dot(self.beta, self.beta)/(self.sigma*self.sigma) * 0.5
        posterior = prior + likelihood
        out = -posterior
        return(out)

    def log_p_y_given_theta(self, observed_point, posterior_point):
        self.load_point(posterior_point)
        X = Variable(observed_point["input"], requires_grad=False).type(self.precision_type)
        y = Variable(observed_point["target"], requires_grad=False).type(self.precision_type)
        #print(self.beta.type)
        #exit()
        num_ob = X.shape[0]
        likelihood = torch.dot(self.beta, torch.mv(torch.t(X), y)) - \
                     torch.sum(logsumexp_torch(Variable(torch.zeros(num_ob)), torch.mv(X, self.beta)))

        return(likelihood.data[0])
    def predict(self,test_samples):
        X = torch.from_numpy(test_samples)
        out = torch.zeros(X.shape[0],2)
        out[:,1] = (torch.sigmoid(torch.mv(X, self.beta.data)))
        out[:,0] = 1-out[:,1]
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