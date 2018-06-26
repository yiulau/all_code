import numpy
from post_processing.ESS_nuts import diagnostics_stan
import pandas as pd

def process_diagnostics(diagnostics_obj,name_list):
    for name in name_list:
        assert name in ("prop_H","accepted","accept_rate","divergent","num_transitions","explode_grad","hit_max_treedepth")
    permuted = diagnostics_obj["permuted"]
    diagnostics = diagnostics_obj["diagnostics"]
    if permuted:
        store = numpy.zeros((len(diagnostics),len(name_list)))
        for i in range(len(diagnostics)):
            for j in range(len(name_list)):
                store[i,j] = diagnostics[i][name_list[j]]
    else:
        # store is num_chains x num mcmc_samples per chain x len(name_list)
        store = numpy.zeros((len(diagnostics),len(diagnostics[0]),len(name_list)))
        for i in range(len(diagnostics)):
            for j in range(len(diagnostics[i])):
                for k in range(len(name_list)):
                    store[i,j,k] = diagnostics[i][j][name_list[k]]
    return(store)

def energy_diagnostics(diagnostics_obj):
    assert diagnostics_obj["permuted"] == False
    # return bfmi-e for each chain
    # return ess,rhat ,posterior mean and sd for energy
    return()

def average_diagnostics(diagnostics_obj,statistic_name):
    # input should be output from mcmc_sampler.get_diagnostics
    # when permuted = True returns statistic separately for each chain
    # when permuted = False returns statistic for the combined chain

    assert statistic_name in ("accepted","accept_rate","divergent","num_transitions","explode_grad","hit_max_treedepth")
    diagnostics = diagnostics_obj["diagnostics"]
    if diagnostics_obj["permuted"]==True:
        sum = 0
        total_terms = 0
        for i in range(len(diagnostics)):
            sum += diagnostics[i][statistic_name]
            total_terms += 1

        out = sum/total_terms
    else:
        sum = 0
        total_terms = 0
        for i in range(len(diagnostics)):
            for j in range(len(diagnostics[i])):
                sum += diagnostics[i][j][statistic_name]
                total_terms += 1

        out = sum / total_terms
    return(out)

# used in float vs double experiments
# find if the chains correspond to same distribution wtih Gelman Rhat statistics
def get_short_diagnostics(mcmc_samples_tensor):
    # return min ESS, percent of parameters Rhat <1.1, ESJD
    full_diagnostics = diagnostics_stan(mcmc_samples_tensor)
    ess = full_diagnostics["ess"]
    min_ess  = min(ess)
    rhat_vec = full_diagnostics["rhat"]
    percent_rhat = sum(rhat_vec<1.1)/(len(rhat_vec))
    out = {"min_ess":min_ess,"percent_rhat":percent_rhat}
    return(out)

