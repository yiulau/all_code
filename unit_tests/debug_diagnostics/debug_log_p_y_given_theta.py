# check that log p_y_given_theta, when exponentiated gives number in the (0,1) range
import torch
from torch.autograd import Variable
from explicit.general_util import logsumexp_torch
from input_data.convert_data_to_dict import get_data_dict

precision_type = "torch.FloatTensor"

beta = Variable(torch.randn(7))

pima_data = get_data_dict("pima_indian")

inputX =  torch.from_numpy(pima_data["input"][0:10,:])
inputy = torch.from_numpy(pima_data["target"][0:10])

input_data = {"input":inputX,"target":inputy}

def log_p_y_given_theta(observed_point):

    X = Variable(observed_point["input"], requires_grad=False).type(precision_type)
    y = Variable(observed_point["target"], requires_grad=False).type(precision_type)
    num_ob = X.shape[0]
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(logsumexp_torch(Variable(torch.zeros(num_ob)), torch.mv(X, beta)))

    return (likelihood.data[0])


out = log_p_y_given_theta(observed_point=input_data)

print(out)