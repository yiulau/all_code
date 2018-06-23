import torch,numpy,os
import torch.nn as nn
from abstract.abstract_class_V import V
from torch.autograd import Variable
import pandas as pd
from torch.autograd import Variable, Function
from explicit.general_util import logsumexp_torch
from distributions.neural_nets.util import gamma_density
def generate_V_fc(input_npdata,precision_type):
    assert precision_type in ("torch.DoubleTensor","torch.FloatTensor")
    class V_fc_hyper(V):
        def __init__(self):
            self.y_np = input_npdata["y_np"]
            self.X_np = input_npdata["X_np"]
            super(V_fc_hyper, self).__init__()
        def V_setup(self):
            self.dim = self.X_np.shape[1]
            self.num_ob = self.X_np.shape[0]
            self.explicit_gradient = True
            self.need_higherorderderiv = True
            self.num_units = 10

            self.hidden_in = nn.Parameter(torch.zeros(self.num_units,self.dim),requires_grad=True)
            self.hidden_in_log_sigma = nn.Parameter(torch.zeros(1),requires_grad=True)
            self.hidden_out = nn.Parameter(torch.zeros(2,self.num_units),requires_grad=True)
            self.hidden_out_log_sigma = nn.Parameter(torch.zeros(1),requires_grad=True)
            self.y = Variable(torch.from_numpy(self.y_np),requires_grad=False).type("torch.LongTensor")
            self.X = Variable(torch.from_numpy(self.X_np),requires_grad=False).type(precision_type)
            # include

            return()

        def forward(self):

            sigmoid = torch.nn.Sigmoid()
            hidden_units = sigmoid((self.hidden_in.mm(self.X.t())))
            out_units = self.hidden_out.mm(hidden_units).t()

            #criterion = nn.NLLLoss()
            criterion = nn.CrossEntropyLoss()
            neg_log_likelihood = criterion(out_units,self.y)
            in_sigma = torch.exp(self.hidden_in_log_sigma)
            out_sigma = torch.exp(self.hidden_out_log_sigma)
            hidden_in_out = -(self.hidden_in * self.hidden_in).sum()*0.5/(in_sigma*in_sigma) - 2*self.hidden_in_log_sigma.sum()
            # print(self.hidden_out_log_sigma)
            # print(self.hidden_in_log_sigma)
            hidden_out_out = -(self.hidden_out * self.hidden_out).sum()*0.5/(out_sigma*out_sigma) - 2*self.hidden_out_log_sigma.sum()
            in_sigma_out = gamma_density(in_sigma,1,1)
            out_sigma_out = gamma_density(out_sigma,1,1)
            #print("likelihood {}".format(likelihood))
            #print("hidden_in_out {}".format(hidden_in_out))
            # print("hidden_out_out {}".format(hidden_out_out))
            # print("in sigma out {}".format(in_sigma_out))
            # print("out sigma {}".format(out_sigma_out))
            prior = hidden_in_out + hidden_out_out + in_sigma_out + out_sigma_out
            #print("prior {}".format(prior))
            #print("neg_loglikelihood {}".format(neg_log_likelihood))
            neg_logposterior = -prior  + neg_log_likelihood
            out = neg_logposterior
            print("hidden in {} ".format(self.hidden_in))
            print("hidden_out {}".format(self.hidden_out))
            print("sigma out {}".format(out_sigma))
            print("sigma in {}".format(in_sigma))
            return(out)

        def predict(self):
            sigmoid = torch.nn.Sigmoid()
            hidden_units = sigmoid((self.hidden_in.mm(self.X.t())))
            out_units = self.hidden_out.mm(hidden_units).t()
            softmax = nn.Softmax()
            prob = softmax(out_units)
            return(prob[:,1].data)