import dill as pickle
import numpy
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics
store_name = 'rhs_toy_sampler.pkl'

num_p = 100
non_zero_p = 20

seedid = 33034
numpy.random.seed(seedid)
true_p = numpy.zeros(num_p)
true_p[:non_zero_p] = numpy.random.randn(non_zero_p)*5
sampler1 = pickle.load(open(store_name, 'rb'))
mcmc_samples_beta = sampler1.get_samples_alt(prior_obj_name="beta",permuted=False)
samples = mcmc_samples_beta["samples"]
w_indices = mcmc_samples_beta["indices_dict"]["w"]
posterior_mean = numpy.mean(samples[:,:,w_indices].reshape(-1,len(w_indices)),axis=0)
print(posterior_mean[:non_zero_p])
print(true_p[:non_zero_p])

#print(mcmc_samples_beta["indices_dict"])

out = sampler1.get_diagnostics(permuted=False)


#processed_diag = process_diagnostics(out,name_list=["accepted"])
#print(processed_diag.shape)

#processed_energy = process_diagnostics(out,name_list=["prop_H"])

print(energy_diagnostics(diagnostics_obj=out))