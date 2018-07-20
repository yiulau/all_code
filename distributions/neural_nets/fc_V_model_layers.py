import torch,numpy,os,math
import torch.nn as nn
from distributions.bayes_model_class import bayes_model_class
from torch.autograd import Variable
from distributions.neural_nets.priors.prior_util import prior_generator

# standard normal prior for input to hidden units
# standard normal prior for hidden to output units

class V_fc_model_layers(bayes_model_class):
    def __init__(self,input_data,precision_type,prior_dict,model_dict):
        self.prior_dict = prior_dict
        self.model_dict = model_dict
        super(V_fc_model_layers, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.input_data["input"].shape[1]
        self.num_ob = self.input_data["target"].shape[0]
        self.num_classes = len(numpy.unique(self.input_data["target"]))
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.num_units = 20
        self.num_layers = self.model_dict["num_layers"]
        prior_fn = prior_generator("normal")
        self.hidden_in = prior_fn(obj=self, name="hidden_in", shape=(self.num_units, self.dim), global_scale=1)
        self.hidden_out = prior_fn(obj=self, name="hidden_out", shape=(self.num_classes, self.num_units),
                                       global_scale=1)
        self.dict_parameters = {"hidden_in": self.hidden_in,"hidden_out":self.hidden_out}
        for i in range(self.num_layers-1):
            prior_hidden_fn = prior_generator(self.prior_dict["name"])
            weights_name = "hidden_{}".format(i+1)
            obj = prior_hidden_fn(obj=self,name=weights_name,shape=(self.num_units,self.num_units),global_scale=1)
            setattr(self,weights_name,obj)
            self.dict_parameters.update({weights_name:obj})



        #self.hidden_in_z = nn.Parameter(torch.zeros(self.num_units, self.dim), requires_grad=True)
        #self.hidden_out_z = nn.Parameter(torch.zeros(2,self.num_units),requires_grad=True)


        self.y = Variable(torch.from_numpy(self.input_data["target"]),requires_grad=False).type("torch.LongTensor")
        self.X = Variable(torch.from_numpy(self.input_data["input"]),requires_grad=False).type(self.precision_type)


        # include

        return()

    def forward(self):
        #print(self.hidden_in.get_val())
        prior = 0
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(self.X.t())))
        for i in range(self.num_layers-1):
            hidden_obj = getattr(self,"hidden_{}".format(i+1))
            relu = torch.nn.ReLU()
            hidden_units = relu(hidden_obj.get_val().mm(hidden_units))
            prior += hidden_obj.get_out()


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
        #print("hidden_out_out {}".format(hidden_out_out))
        #print("in sigma out {}".format(in_sigma_out))
        # print("out sigma {}".format(out_sigma_out))
        prior += hidden_in_out + hidden_out_out #+ in_sigma_out + out_sigma_out
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
        for i in range(self.num_layers-1):
            hidden_obj = getattr(self,"hidden_{}".format(i+1))
            hidden_units = torch.tanh(hidden_obj.get_val().mm(hidden_units))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        softmax = nn.Softmax(dim=-1)
        prob = softmax(out_units)
        return(prob.data)

    def log_p_y_given_theta(self, observed_point, posterior_point):
        self.load_point(posterior_point)
        X = Variable(observed_point["input"]).type(self.precision_type)
        y = Variable(observed_point["target"]).type("torch.LongTensor")
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        for i in range(self.num_layers-1):
            hidden_obj = getattr(self,"hidden_{}".format(i+1))
            relu = torch.nn.ReLU
            hidden_units = relu(hidden_obj.get_val().mm(hidden_units))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        criterion = nn.CrossEntropyLoss()
        neg_log_likelihood = criterion(out_units, y)
        out = -neg_log_likelihood
        return(out)