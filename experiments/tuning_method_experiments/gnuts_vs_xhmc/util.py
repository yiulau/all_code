from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics
from post_processing.ESS_nuts import ess_stan
import numpy
def fun_extract_median_ess(sampler):
    # median ess
    # remove failed chains first
    sampler.remove_failed_chains()
    full_mcmc_tensor = get_params_mcmc_tensor(sampler=sampler)
    ess = ess_stan(full_mcmc_tensor)
    median_ess = numpy.median(ess)
    return([median_ess],["median_ess"])


def get_diagnostics(sampler):
    #
    # want to return numpy array of dimensions [num_chains,9]
    # as well as list of column names

    feature_names = ["num_restarts","num_divergent","num_hit_max_tree_depth","ave_num_transitions","bfmi","lp_ess","lp_rhat","difficulty"]
    feature_names = ["num_chains_removed"]

    sampler.remove_failed_chains()
    out = sampler.get_diagnostics(permuted=False)
    num_restarts = sampler.metadata.num_restarts
    num_chains_removed = sampler.metadata.num_chains_removed
    processed_diag = process_diagnostics(out, name_list=["divergent"])
    num_divergent = processed_diag.sum(axis=1)
    processed_diag = process_diagnostics(out, name_list=["hit_max_tree_depth"])
    hix_max_tree_depth = processed_diag.sum(axis=1)
    processed_diag = process_diagnostics(out, name_list=["num_transitions"])
    ave_num_transitions = processed_diag.mean(axis=1)
    energy_summary = energy_diagnostics(diagnostics_obj=out)
    mixed_mcmc_tensor = sampler.get_samples(permuted=True)
    mcmc_cov = numpy.cov(mixed_mcmc_tensor, rowvar=False)
    mcmc_sd_vec = numpy.sqrt(numpy.diagonal(mcmc_cov))
    difficulty = max(mcmc_sd_vec) / min(mcmc_sd_vec)
    num_id = sampler.num_chains
    output = numpy.zeros(num_id,len(feature_names))

    output[:,0] = num_restarts
    output[:,1] = num_divergent
    output[:,2] = hix_max_tree_depth
    output[:,3] = ave_num_transitions
    output[:,4] = energy_summary["bfmi_list"]
    output[:,5] = energy_summary["ess"]
    output[:,6] = energy_summary["rhat"]
    output[:,7] = difficulty
    output[:,8] = num_chains_removed
    return(output,feature_names)


