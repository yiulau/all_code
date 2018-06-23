import torch
from distributions.bayes_model_class import bayes_model_class
from torch.autograd import Variable
from distributions.neural_nets.priors.prior_util import prior_generator
from explicit.general_util import logsumexp_torch
from distributions.neural_nets.priors.horseshoe_1 import horseshoe_1

class V_hs_lr(bayes_model_class):
    def __init__(self,input_data,precision_type):

        super(V_hs_lr, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.input_data["input"].shape[1]
        self.num_ob = self.input_data["input"].shape[0]
        self.explicit_gradient = False
        self.need_higherorderderiv = False
        hs_prior_generator_fun = prior_generator("horseshoe_3")
        prior_obj = hs_prior_generator_fun(obj=self,name="beta",shape=[self.dim])
        #self.beta_obj = prior_obj.generator(obj=self,name="beta",shape=[self.dim])
        self.beta_obj = prior_obj
        self.y = Variable(torch.from_numpy(self.input_data["target"]),requires_grad=False).type(self.precision_type)
        self.X = Variable(torch.from_numpy(self.input_data["input"]),requires_grad=False).type(self.precision_type)
        # include
        #self.sigma =1

        return()

    def forward(self):
        beta = self.beta_obj.get_val()

        likelihood = -((self.y - torch.mv(self.X,beta))*(self.y-torch.mv(self.X,beta))).sum()*0.5
        prior = self.beta_obj.get_out()
        posterior = prior + likelihood
        out = -posterior
        return(out)
