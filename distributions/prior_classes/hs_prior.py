from distributions.prior_classes.prior_base_class import prior_class
import torch
import torch.nn as nn

class hs_prior(prior_class):
    def __init__(self):
        super(hs_prior, self).__init__()
        pass

    def create_hyper_par_fun(self):
        self.hyper_param_blocks = []
        for param in self.V.prior_param_blocks:
            name = param.name
            shape = param.shape
            r1_local = nn.Parameter(torch.zeros(shape), requires_grad=True)
            r1_global = nn.Parameter(torch.zeros(1),requires_grad=True)
            r2_
            setattr(self.V,name+"_r1_local",r1_local)
            hyper_param_dict = {"r1_local":r1_local}
            setattr(self.V, name + "_r1_global", r1_global)
            hyper_param_dict = {"r1_global": r1_global}
            self.hyper_param_blocks.append({"original_param":param,"hyperparam_dict":hyper_param_dict})


    def prior_forward(self):
        out = 0
        for i in range(len(self.V.prior_param_blocks)):
            log_sigma= self.hyper_param_blocks["hyperparam_dict"]["log_sigma"]
            sigma = torch.exp(log_sigma)
            out += -(self.V.prior_param_blocks[i] * self.V.prior_param_blocks[i]/sigma).sum() * 0.5 + log_sigma
        return (out)