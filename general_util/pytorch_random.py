import torch,numpy
#from torch.distributions.gamma import Gamma

def generate_gamma(alpha,beta):
    alpha_np = alpha.numpy()
    beta_np = beta.numpy()
    out_np = numpy.random.gamma(shape=alpha_np,scale=1/beta_np)
    out = torch.from_numpy(out_np).type(alpha.dtype)
    #m = Gamma(concentration=alpha,rate=beta)
    #out = m.sample()
    return(out)

def log_gamma_density(x,alpha,beta):
    out = -(alpha+1)*torch.log(x) - beta/x
    return(out)
# alpha_tensor = torch.tensor([0.1,1,3.3])
# beta_tensor = torch.tensor([0.1,13,12])
# #
# out = generate_gamma(alpha_tensor,beta_tensor)
# print(out)