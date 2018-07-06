from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics,get_params_mcmc_tensor,get_short_diagnostics
import numpy
def fun_extract_ess(sampler):

    # remove failed chains first
    full_mcmc_tensor = get_params_mcmc_tensor(sampler=sampler)

    diagnostics,col_names = get_diagnostics(sampler)
    return()


def get_diagnostics(sampler):
    #
    # want to return numpy array of dimensions [num_chains,3+3+5+1]
    # as well as list of column names
    out = sampler.get_diagnostics(permuted=False)
    processed_diag = process_diagnostics(out, name_list=["divergent"])
    print(processed_diag.sum(axis=1))
    processed_diag = process_diagnostics(out, name_list=["hit_max_tree_depth"])
    hix_max_tree_depth = processed_diag.sum(axis=1)
    processed_diag = process_diagnostics(out, name_list=["num_transitions"])
    ave_num_transitions = processed_diag.mean(axis=1)
    print(energy_diagnostics(diagnostics_obj=out))
    mixed_mcmc_tensor = sampler.get_samples(permuted=True)

    mcmc_cov = numpy.cov(mixed_mcmc_tensor, rowvar=False)
    mcmc_sd_vec = numpy.sqrt(numpy.diagonal(mcmc_cov))

    print(max(mcmc_sd_vec) / min(mcmc_sd_vec))
    return()


