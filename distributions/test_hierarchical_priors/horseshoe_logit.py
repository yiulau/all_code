import torch
from distributions.bayes_model_class import bayes_model_class
from torch.autograd import Variable
from distributions.neural_nets.priors.prior_util import prior_generator
from explicit.general_util import logsumexp_torch


class V_logistic_regression_hs(bayes_model_class):
    def __init__(self,input_data,precision_type):

        super(V_logistic_regression_hs, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.X_np.shape[1]
        self.num_ob = self.X_np.shape[0]
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        prior_generator_fn = prior_generator("horseshoe_3")
        prior_obj = prior_generator_fn(obj=self,name="beta",shape=[self.dim])
        #self.beta_obj = prior_obj.generator(obj=self,name="beta",shape=[self.dim])
        self.beta_obj = prior_obj
        self.y = Variable(torch.from_numpy(self.input_data["target"]),requires_grad=False).type(self.precision_type)
        self.X = Variable(torch.from_numpy(self.input_data["input"]),requires_grad=False).type(self.precision_type)
        self.dict_parameters = {"beta":self.beta_obj}

        return()

    def forward(self):
        beta = self.beta_obj.get_val()
        likelihood = torch.dot(beta, torch.mv(torch.t(self.X), self.y)) - \
                     torch.sum(logsumexp_torch(Variable(torch.zeros(self.num_ob)), torch.mv(self.X, beta)))
        prior = self.beta_obj.get_out()
        posterior = prior + likelihood
        out = -posterior
        return(out)