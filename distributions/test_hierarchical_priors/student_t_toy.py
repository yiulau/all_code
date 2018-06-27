from distributions.bayes_model_class import bayes_model_class
from distributions.neural_nets.priors.prior_util import prior_generator
from torch.autograd import Variable
import torch

class V_student_toy(bayes_model_class):
    def __init__(self,input_data,precision_type):

        super(V_student_toy, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = len(self.input_data["target"])
        self.explicit_gradient = False
        self.need_higherorderderiv = False
        hs_prior_generator_fun = prior_generator("gaussian_inv_gamma_2")
        prior_obj = hs_prior_generator_fun(obj=self,name="beta",shape=[self.dim])
        self.beta_obj = prior_obj
        self.y = Variable(torch.from_numpy(self.input_data["target"]),requires_grad=False).type(self.precision_type)
        self.dict_parameters = {"beta":self.beta_obj}

        return()

    def forward(self):
        beta = self.beta_obj.get_val()
        likelihood = -((self.y - beta)*(self.y-beta)).sum()*0.5
        prior = self.beta_obj.get_out()
        posterior = prior + likelihood
        out = -posterior
        return(out)