import dill as pickle
import numpy
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics
sampler1 = pickle.load(open('temp_save_sampler1.pkl', 'rb'))

mcmc_samples_beta = sampler1.get_samples_alt(prior_obj_name="beta",permuted=False)
samples = mcmc_samples_beta["samples"]
posterior_mean = numpy.mean(samples.reshape(-1,101),axis=0)
out = sampler1.get_diagnostics(permuted=False,include_warmup=True)


processed_diag = process_diagnostics(out,name_list=["num_transitions"])
print(processed_diag[0,0:1000,0])
