def combine_chains(mcmc_tensor):
    assert len(mcmc_tensor.shape)==3

    combined_mcmc_tensor = mcmc_tensor.view(-1,mcmc_tensor.shape[2]).clone()
    return(combined_mcmc_tensor)