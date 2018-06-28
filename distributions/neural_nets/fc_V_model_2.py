import torch,numpy,os,math
import torch.nn as nn
from distributions.bayes_model_class import bayes_model_class
from torch.autograd import Variable
from distributions.neural_nets.priors.prior_util import prior_generator

# hierarchical prior for hidden to output units, scale = sqrt(1/num_hidden_units)
#  normal prior for input to hidden units with variance = (1/input_dim)
class V_fc_model_2(bayes_model_class):
    def __init__(self,input_data,precision_type,prior_dict,model_dict):
        self.prior_dict = prior_dict
        self.model_dict = model_dict
        super(V_fc_model_2, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.input_data["input"].shape[1]
        self.num_ob = self.input_data["target"].shape[0]
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.num_units = self.model_dict["num_units"]
        prior_hidden_fn = prior_generator(self.prior_dict["name"])
        prior_out_fn = prior_generator("normal")
        self.hidden_in = prior_hidden_fn(obj=self,name="hidden_in",shape=(self.num_units,self.dim),global_scale=math.sqrt(1/self.dim))
        self.hidden_out = prior_out_fn(obj=self,name="hidden_out",shape=(2,self.num_units),var=1/self.num_units)

        #self.hidden_in_z = nn.Parameter(torch.zeros(self.num_units, self.dim), requires_grad=True)
        #self.hidden_out_z = nn.Parameter(torch.zeros(2,self.num_units),requires_grad=True)


        self.y = Variable(torch.from_numpy(self.input_data["target"]),requires_grad=False).type("torch.LongTensor")
        self.X = Variable(torch.from_numpy(self.input_data["input"]),requires_grad=False).type(self.precision_type)
        self.dict_parameters = {"hidden_in": self.hidden_in,"hidden_out":self.hidden_out}
        # include

        return()

    def forward(self):

        hidden_units = torch.tanh((self.hidden_in.get_val().mm(self.X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        #print(out_units.shape)
        #print(out_units)
        #exit()
        #criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
        #print(self.y)
        #exit()
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
        # print("hidden in {} ".format(self.hidden_in))
        # print("hidden_out {}".format(self.hidden_out))
        # print("sigma out {}".format(self.hidden_out.get_val()))
        # print("sigma in {}".format(self.hidden_out.get_val()))
        return(out)

    def predict(self,inputX):
        X = Variable(torch.from_numpy(inputX),requires_grad=False).type(self.precision_type)
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        softmax = nn.Softmax()
        prob = softmax(out_units)
        return(prob.data)

    def log_p_y_given_theta(self, observed_point, posterior_point):
        self.load_point(posterior_point)
        X = observed_point["input"]
        y = observed_point["target"]
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        criterion = nn.CrossEntropyLoss()
        neg_log_likelihood = criterion(out_units, self.y)
        out = -neg_log_likelihood
        return(out)