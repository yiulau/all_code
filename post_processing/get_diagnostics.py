import numpy
def get_diagnostics_from_sample(diagnostics_obj,permuted,name):
    assert name in ("prop_H","accepted","accept_rate","divergent","num_transitions","explode_grad")
    if permuted:
        store = [None]*len(diagnostics_obj)
        for i in range(len(diagnostics_obj)):
            store[i] = diagnostics_obj[i]["name"]

    else:
        total_len = len(diagnostics_obj[0]) * len(diagnostics_obj)
        # store is num_chains x num mcmc_samples per chain x 1
        store = numpy.zeros(len(diagnostics_obj),len(diagnostics_obj[0]),1)
        for i in range(len(diagnostics_obj)):
            for j in range(len(diagnostics_obj[i])):
                store[i,j,0] = diagnostics_obj[i][j]["name"]
    return(store)