import dill as pickle
import numpy
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics

seed = 1
numpy.random.seed(seed)
non_zero_num_p = 20
full_p = 400
num_samples = 100
X_np = numpy.random.randn(num_samples,full_p)*5
true_beta = numpy.zeros(full_p)
true_beta[:non_zero_num_p] = numpy.random.randn(non_zero_num_p)*5
y_np = X_np.dot(true_beta) + numpy.random.randn(num_samples)

store_name = 'hs_lr_sampler.pkl'
sampler1 = pickle.load(open(store_name, 'rb'))

mcmc_samples_beta = sampler1.get_samples_alt(prior_obj_name="beta",permuted=False)
#print(mcmc_samples_beta["indices_dict"])
#exit()

samples = mcmc_samples_beta["samples"]
w_indices = mcmc_samples_beta["indices_dict"]["w"]
print(samples.shape)
posterior_mean = numpy.mean(samples[:,:,w_indices].reshape(-1,len(w_indices)),axis=0)
print(posterior_mean[:non_zero_num_p])
print(true_beta[:non_zero_num_p])

#print(mcmc_samples_beta["indices_dict"])

out = sampler1.get_diagnostics(permuted=False)


#processed_diag = process_diagnostics(out,name_list=["accepted"])
#print(processed_diag.shape)

#processed_energy = process_diagnostics(out,name_list=["prop_H"])

print(energy_diagnostics(diagnostics_obj=out))