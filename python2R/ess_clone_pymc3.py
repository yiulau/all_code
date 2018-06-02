import numpy as np
def get_neff(x):
    """Compute the effective sample size for a 2D array
    """
    trace_value = x.T
    nchain, n_samples = trace_value.shape

    acov = np.asarray([autocov(trace_value[chain]) for chain in range(nchain)])

    chain_mean = trace_value.mean(axis=1)
    chain_var = acov[:, 0] * n_samples / (n_samples - 1.)
    acov_t = acov[:, 1] * n_samples / (n_samples - 1.)
    mean_var = np.mean(chain_var)
    var_plus = mean_var * (n_samples - 1.) / n_samples
    var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_samples)
    rho_hat_even = 1.
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1. - (mean_var - np.mean(acov_t)) / var_plus
    rho_hat_t[1] = rho_hat_odd
    # Geyer's initial positive sequence
    max_t = 1
    t = 1
    while t < (n_samples - 2) and (rho_hat_even + rho_hat_odd) >= 0.:
        rho_hat_even = 1. - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1. - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        max_t = t + 2
        t += 2

    # Geyer's initial monotone sequence
    t = 3
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2
    ess = nchain * n_samples
    ess = ess / (-1. + 2. * np.sum(rho_hat_t))
    return ess