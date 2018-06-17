import numpy,torch
from abstract.abstract_class_point import point

from post_processing.util import combine_chains
# bfmi-e
import math
def bfmi_e(vector_of_energy):
    e_bar = numpy.mean(vector_of_energy)
    denom = numpy.square(vector_of_energy - e_bar).sum()
    numerator = numpy.square(vector_of_energy[1:]-vector_of_energy[:(len(vector_of_energy)-1)]).sum()
    out = numerator/denom
    return(out)


def lpd(p_y_given_theta,posterior_samples,observed_data):
    S = len(posterior_samples)
    n = len(observed_data)
    torch_input = torch.from_numpy(observed_data["input"])
    torch_target = torch.from_numpy(observed_data["target"])
    out = 0
    for i in range(n):
        temp = numpy.zeros(S)
        observed_point = {"input":torch_input[i:i+1,:],"target":torch_target[i:i+1]}
        for j in range(S):
            temp[j]=p_y_given_theta(observed_point,posterior_samples[j])
        out += math.log(sum(temp)/S)
    return(out)

# def lpd_efficient(p_y_given_theta,posterior_samples,observed_data):
#     return()
#
# def pwaic_efficient(log_p_y_given_theta,posterior_samples,observed_data):
#     return()
def pwaic(log_p_y_given_theta,posterior_samples,observed_data):
    S = len(posterior_samples)
    n = len(observed_data["input"].shape[0])
    torch_input = torch.from_numpy(observed_data["input"])
    torch_target = torch.from_numpy(observed_data["target"])
    out = 0
    for i in range(n):
        temp = numpy.zeros(S)
        observed_point = {"input":torch_input[i:i+1,:],"target":torch_target[i:i+1]}
        for j in range(S):
            temp[j]= log_p_y_given_theta(observed_point, posterior_samples[j])
        out += numpy.var(temp)
    return (out)

def WAIC(posterior_samples,observed_data,V):
    # posterior samples a list of len num_samples each containing point _obj
    # observed_data a dict with "input" and "target" keys containing np array
    elpd = lpd(V.p_y_given_theta,posterior_samples,observed_data) - pwaic(V.log_p_y_given_theta,posterior_samples,observed_data)
    out = -2*elpd
    return(out)


def elpd_waic_vec(posterior_samples,observed_data,V):
    n = len(observed_data["input"].shape[0])
    S = len(posterior_samples)
    store_waic_i_vec = numpy.zeros(n)
    torch_input = torch.from_numpy(observed_data["input"])
    torch_target = torch.from_numpy(observed_data["target"])
    for i in range(n):
        observed_point = {"input":torch_input[i:i+1,:],"target":torch_target[i:i+1]}
        temp_lpd = numpy.zeros(S)
        temp_pwaic = numpy.zeros(S)
        for j in range(S):
            temp_lpd[j]= V.p_y_given_theta(observed_point,posterior_samples[j])
            temp_pwaic[j] = V.log_p_y_given_theta(observed_point, posterior_samples[j])
        store_waic_i_vec = math.log(sum(temp_lpd)/S) - numpy.var(temp_pwaic)

    return(store_waic_i_vec)

def se_elpd_waic(elpd_waic_vec):
    n = len(elpd_waic_vec)
    out = math.sqrt(n*numpy.var(elpd_waic_vec))
    return(out)

def convert_mcmc_tensor_to_list_points(mcmc_tensor,v_obj):
    num_samples = mcmc_tensor.shape[0]
    out = [None]*num_samples
    for i in range(num_samples):
        this_point = point(V=v_obj)
        this_point.flattened_tensor.copy_(mcmc_tensor[i,:])
        this_point.load_flatten()
        out[i] = this_point
    return(out)


def esjd(mcmc_tensor):
    # if more than one chain, combine into one long chain
    if len(mcmc_tensor.shape)>2:
        combined_mcmc_tensor = combine_chains(mcmc_tensor)
    else:
        combined_mcmc_tensor = mcmc_tensor


