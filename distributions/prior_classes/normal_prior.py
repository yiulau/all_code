from distributions.prior_classes.prior_base_class import prior_class

class normal_prior(prior_class):
    def __init__(self):
        super(normal_prior, self).__init__()
        pass

    def create_hyper_par_fun(self):
        pass

    def prior_forward(self):
        out = 0
        for i in range(self.V.num_var):
            out += -(self.V.list_var[i] * self.V.list_var[i]).sum() * 0.5
        return(out)

