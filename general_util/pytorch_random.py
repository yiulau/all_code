import torch
from torch.distributions.gamma import Gamma

def generate_gamma(alpha,beta):
    m = Gamma(concentration=alpha,rate=beta)
    out = m.sample()
    return(out)


# alpha_tensor = torch.tensor([0.1,1,3.3])
# beta_tensor = torch.tensor([0.1,13,12])
#
# out = generate_gamma(alpha_tensor,beta_tensor)
# print(out)