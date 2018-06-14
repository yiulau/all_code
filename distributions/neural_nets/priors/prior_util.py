from distributions.neural_nets.priors.horseshoe_1 import horseshoe_1
from distributions.neural_nets.priors.horseshoe_2 import horseshoe_2
from distributions.neural_nets.priors.horseshoe_3 import horseshoe_3
from distributions.neural_nets.priors.horseshoe_4 import horseshoe_4
from distributions.neural_nets.priors.horseshoe_ard import horseshoe_ard
from distributions.neural_nets.priors.gaussian_inv_gamma_1 import gaussian_inv_gamma_1
from distributions.neural_nets.priors.gaussian_inv_gamma_2 import gaussian_inv_gamma_2
def prior_generator(name):
    if name=="horseshoe_1":
        out = horseshoe_1
    elif name=="horseshoe_2":
        out = horseshoe_2

    elif name=="horseshoe_3":
        out = horseshoe_3

    elif name=="horseshoe_4":
        out = horseshoe_4

    elif name=="horseshoe_ard":
        out = horseshoe_ard

    elif name=="gaussian_inv_gamma_1":
        out = gaussian_inv_gamma_1

    elif name=="gaussian_inv_gamma_2":
        out = gaussian_inv_gamma_2
    return(out)


#out = prior_generator("horseshoe_1")