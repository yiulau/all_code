import torch,numpy,os
import torch.nn as nn
from distributions.bayes_model_class import bayes_model_class
from torch.autograd import Variable


# precision_type = 'torch.DoubleTensor'
# #precision_type = 'torch.FloatTensor'
# torch.set_default_tensor_type(precision_type)
from distributions.neural_nets.priors.prior_util import prior_generator
# horseshoe prior for hidden to out units, scale = 1/num_hidden_units
# standard normal prior for input to hidden units with variance 1/N_input
class V_fc_student(bayes_model_class):
    def __init__(self,input_data,precision_type):
        super(V_fc_student, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.input_data["input"].shape[1]
        self.num_ob = self.input_data["target"].shape[0]
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.num_units = 10
        prior_hidden_fn = prior_generator("gaussian_inv_gamma_2")
        prior_out_fn = prior_generator("normal",var=1/self.num_units)
        self.hidden_in = prior_hidden_fn(obj=self,name="hidden_in",shape=(self.num_units,self.dim))
        self.hidden_out = prior_out_fn(obj=self,name="hidden_out",shape=(2,self.num_units),global_scale=1/self.num_units)
        #self.hidden_in_z = nn.Parameter(torch.zeros(self.num_units, self.dim), requires_grad=True)
        #self.hidden_out_z = nn.Parameter(torch.zeros(2,self.num_units),requires_grad=True)


        self.y = Variable(torch.from_numpy(self.y_np),requires_grad=False).type("torch.LongTensor")
        self.X = Variable(torch.from_numpy(self.X_np),requires_grad=False).type(self.precision_type)
        # include
        self.dict_parameters = {"hidden_in":self.hidden_in,"hidden_out":self.hidden_out}


        return()

    def forward(self):

        hidden_units = torch.tanh((self.hidden_in.get_val().mm(self.X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()

        #criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
        neg_log_likelihood = criterion(out_units,self.y)
        hidden_in_out = self.hidden_in.get_out()
        hidden_out_out = self.hidden_out.get_out()
      #  in_sigma_out = gamma_density(in_sigma,1,1)
      #  out_sigma_out = gamma_density(out_sigma,1,1)
        #print("likelihood {}".format(likelihood))
        #print("hidden_in_out {}".format(hidden_in_out))
        # print("hidden_out_out {}".format(hidden_out_out))
        # print("in sigma out {}".format(in_sigma_out))
        # print("out sigma {}".format(out_sigma_out))
        prior = hidden_in_out + hidden_out_out #+ in_sigma_out + out_sigma_out
        #print("prior {}".format(prior))
        #print("neg_loglikelihood {}".format(neg_log_likelihood))
        neg_logposterior = -prior  + neg_log_likelihood
        out = neg_logposterior
        #print("hidden in {} ".format(self.hidden_in))
        #print("hidden_out {}".format(self.hidden_out))
        #print("sigma out {}".format(self.hidden_out.get_val()))
        #print("sigma in {}".format(self.hidden_out.get_val()))
        return(out)

    def predict(self,inputX):
        X = Variable(torch.from_numpy(inputX),requires_grad=False).type(self.precision_type)
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        softmax = nn.Softmax()
        prob = softmax(out_units)
        return(prob[:,1].data)

    def log_p_y_given_theta(self, observed_point, posterior_point):
        # should check that when exponentiated maps to (0,1) i.e.
        # output is negative
        self.load_point(posterior_point)
        X = observed_point["input"]
        y = observed_point["target"]
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        criterion = nn.CrossEntropyLoss()
        neg_log_likelihood = criterion(out_units, self.y)
        out = -neg_log_likelihood
        return(out)