import numpy
def get_diagnostics_from_sample(diagnostics_obj,permuted,name):
    assert name in ("prop_H","accepted","accept_rate","divergent","num_transitions","explode_grad")
    if permuted:
        store = [None]*len(diagnostics_obj)
        for i in range(len(diagnostics_obj)):
            store[i] = diagnostics_obj[i][name]

    else:
        total_len = len(diagnostics_obj[0]) * len(diagnostics_obj)
        # store is num_chains x num mcmc_samples per chain x 1
        store = numpy.zeros((len(diagnostics_obj),len(diagnostics_obj[0]),1))
        for i in range(len(diagnostics_obj)):
            for j in range(len(diagnostics_obj[i])):
                store[i,j,0] = diagnostics_obj[i][j][name]
    return(store)


def percent_diagnostics(diagnostics,statistic_name):
    # input should be output from get_samples_p_diag
    sum = 0
    total_terms = 0
    for i in range(len(diagnostics)):
        for j in range(len(diagnostics[i])):
            sum += diagnostics[i][j][statistic_name]
            total_terms += 1

    out = sum/total_terms
    return(out)

