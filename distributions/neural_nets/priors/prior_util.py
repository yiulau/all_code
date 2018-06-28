from distributions.neural_nets.priors.horseshoe_1 import horseshoe_1
from distributions.neural_nets.priors.horseshoe_2 import horseshoe_2
from distributions.neural_nets.priors.horseshoe_3 import horseshoe_3
from distributions.neural_nets.priors.horseshoe_4 import horseshoe_4
from distributions.neural_nets.priors.horseshoe_ard import horseshoe_ard
from distributions.neural_nets.priors.horseshoe_ard_2 import horseshoe_ard_2
from distributions.neural_nets.priors.rhorseshoe_1 import rhorseshoe_1
from distributions.neural_nets.priors.rhorseshoe_2 import rhorseshoe_2
from distributions.neural_nets.priors.rhorseshoe_3 import rhorseshoe_3
from distributions.neural_nets.priors.rhorseshoe_4 import rhorseshoe_4
from distributions.neural_nets.priors.rhorseshoe_ard import rhorseshoe_ard
from distributions.neural_nets.priors.rhorseshoe_ard_2 import rhorseshoe_ard_2
from distributions.neural_nets.priors.gaussian_inv_gamma_1 import gaussian_inv_gamma_1
from distributions.neural_nets.priors.gaussian_inv_gamma_2 import gaussian_inv_gamma_2
from distributions.neural_nets.priors.gaussian_inv_gamma_ard import gaussian_inv_gamma_ard
from distributions.neural_nets.priors.gaussian_inv_gamma_ard_2 import gaussian_inv_gamma_ard_2
from distributions.neural_nets.priors.standard_normal import standard_normal
from distributions.neural_nets.priors.normal import normal

def prior_generator(name,**kwargs):
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
    elif name=="horseshoe_ard_2":
        out = horseshoe_ard_2

    elif name=="rhorseshoe_1":
        out = rhorseshoe_1
    elif name=="rhorseshoe_2":
        out = rhorseshoe_2

    elif name=="rhorseshoe_3":
        out = rhorseshoe_3

    elif name=="rhorseshoe_4":
        out = rhorseshoe_4

    elif name=="rhorseshoe_ard":
        out = rhorseshoe_ard
    elif name=="rhorseshoe_ard_2":
        out = rhorseshoe_ard_2
    elif name=="gaussian_inv_gamma_1":
        out = gaussian_inv_gamma_1

    elif name=="gaussian_inv_gamma_2":
        out = gaussian_inv_gamma_2

    elif name=="gaussian_inv_gamma_ard":
        out = gaussian_inv_gamma_ard
    elif name=="gaussian_inv_gamma_ard_2":
        out = gaussian_inv_gamma_ard_2

    elif name=="standard_normal":
        out = standard_normal

    elif name=="normal":
        out = normal
    return(out)


#out = prior_generator("horseshoe_1")