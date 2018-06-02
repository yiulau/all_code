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


    def prepare_prior(self):
        # functions to add hyperparamters to parameter list
        self.prior_state = prior_state(self)
        return()
    def forward(self):
        log_posterior = self.log_likelihood() + self.log_prior()
        out = -log_posterior
        return(out)

    def get_beta(self):
        assert hasattr(self, "prior_state")
        out = self.prior_state.get_centered_param()
        return(out)
