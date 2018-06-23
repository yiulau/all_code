import torch,numpy,os
import torch.nn as nn
from distributions.bayes_model_class import bayes_model_class
from torch.autograd import Variable
import pandas as pd
from torch.autograd import Variable, Function
from explicit.general_util import logsumexp_torch
from distributions.neural_nets.util import log_inv_gamma_density
from general_util.pytorch_random import generate_gamma

# for gibbs sampler
# separate inverse gamma prior for weight variances alpha =0.5 ,beta =0.5
# for input-to-hidden and hidden-to-output weights
class V_fc_hierarchical_block(bayes_model_class):
    def __init__(self,input_data,precision_type):
        super(V_fc_hierarchical_block, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.input_data["input"].shape[1]
        self.num_ob = self.input_data["input"].shape[0]
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.num_units = 10

        self.hidden_in = nn.Parameter(torch.zeros(self.num_units,self.dim),requires_grad=True)
        self.hidden_in_sigma2 = Variable(torch.zeros(1),requires_grad=False)
        self.hidden_out = nn.Parameter(torch.zeros(2,self.num_units),requires_grad=True)
        self.hidden_out_sigma2 = Variable(torch.zeros(1),requires_grad=False)
        self.y = Variable(torch.from_numpy(self.y_np),requires_grad=False).type("torch.LongTensor")
        self.X = Variable(torch.from_numpy(self.X_np),requires_grad=False).type(self.precision_type)
        # include
        # the two lists need to match
        self.list_hyperparam = [self.hidden_in_log_sigma,self.hidden_out_log_sigma]
        self.list_param = [self.hidden_in,self.hidden_out]
        return()

    def update_hyperparam(self):
        alpha_tensor = torch.zeros(len(self.list_hyperparam))
        beta_tensor = torch.zeros(len(self.list_hyperparam))
        for i in range(len(self.list_hyperparam)):
            n = len(self.list_param[i].data.view(-1))
            norm = ((self.list_param[i].data)*(self.list_param[i].data)).sum()
            alpha_tensor[i] = n*0.5 + 1
            beta_tensor[i] = norm *0.5 + 1
        new_hyperparam_val = 1/generate_gamma(alpha=alpha_tensor,beta=beta_tensor)
        for i in range(len(self.list_hyperparam)):
            self.list_hyperparam[i].data.copy_(new_hyperparam_val[i])
        return()
    def forward(self):

        sigmoid = torch.nn.Sigmoid()
        hidden_units = sigmoid((self.hidden_in.mm(self.X.t())))
        out_units = self.hidden_out.mm(hidden_units).t()

        #criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
        neg_log_likelihood = criterion(out_units,self.y)
        in_sigma2 = self.hidden_in_sigma2
        out_sigma2 = self.hidden_out_sigma2
        hidden_in_out = -(self.hidden_in * self.hidden_in).sum()*0.5/(in_sigma2)
        # print(self.hidden_out_log_sigma)
        # print(self.hidden_in_log_sigma)
        hidden_out_out = -(self.hidden_out * self.hidden_out).sum()*0.5/(out_sigma2)
        in_sigma2_out = log_inv_gamma_density(x=in_sigma2,alpha=0.5,beta=0.5)
        out_sigma2_out = log_inv_gamma_density(x=out_sigma2,alpha=0.5,beta=0.5)
        #print("likelihood {}".format(likelihood))
        #print("hidden_in_out {}".format(hidden_in_out))
        # print("hidden_out_out {}".format(hidden_out_out))
        # print("in sigma out {}".format(in_sigma_out))
        # print("out sigma {}".format(out_sigma_out))
        prior = hidden_in_out + hidden_out_out + in_sigma2_out + out_sigma2_out
        #print("prior {}".format(prior))
        #print("neg_loglikelihood {}".format(neg_log_likelihood))
        neg_logposterior = -prior  + neg_log_likelihood
        out = neg_logposterior
        print("hidden in {} ".format(self.hidden_in))
        print("hidden_out {}".format(self.hidden_out))
        print("sigma out {}".format(out_sigma2))
        print("sigma in {}".format(in_sigma2))
        return(out)

    def predict(self):
        sigmoid = torch.nn.Sigmoid()
        hidden_units = sigmoid((self.hidden_in.mm(self.X.t())))
        out_units = self.hidden_out.mm(hidden_units).t()
        softmax = nn.Softmax()
        prob = softmax(out_units)
        return(prob[:,1].data)

    def load_hyperparam(self,list_hyperparam):
        for i in range(len(self.list_hyperparam)):
            self.list_hyperparam[i].data.copy_(list_hyperparam[i])
        return()


    def log_p_y_given_theta(self, observed_point, posterior_point):
        self.load_point(posterior_point)
        X = Variable(torch.from_numpy(observed_point["input"]), requires_grad=False).type(self.precision_type)
        y = Variable(torch.from_numpy(observed_point["target"]), requires_grad=False).type(self.precision_type)
        hidden_units = torch.tanh((self.hidden_in.mm(X.t())))
        out_units = self.hidden_out.mm(hidden_units).t()
        criterion = nn.CrossEntropyLoss()
        neg_log_likelihood = criterion(out_units, y)
        out = -neg_log_likelihood
        out = out.data[0]
        return(out)
    def get_hyperparam(self):
        out = []
        for i in range(len(self.list_hyperparam)):
            out.append(self.list_hyperparam[i].data.clone())
        return(out)


