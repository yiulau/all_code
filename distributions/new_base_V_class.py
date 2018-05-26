from abstract.abstract_class_V import V
import abc
class new_base_V_class(V):

    def __init__(self):
        super(new_base_V_class, self).__init__()

    @abc.abstract_method
    def log_likelihood(self):
        return()
    @abc.abstract_method
    def log_prior(self):
        return()
    @abc.abstract_method
    def prepare_prior(self):
        # functions to add hyperparamters to parameter list
        pass
    def forward(self):
        out = self.log_likelihood() + self.log_prior()
        return(out)

    def get_beta(self):
        assert hasattr(self, "prior_state")
        out = self.prior_state.get_centered_param()
        return(out)
