import dill as pickle
import numpy
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics

store_name = 'hs3_fc1_sampler.pkl'
sampler1 = pickle.load(open(store_name, 'rb'))

mcmc_samples_hidden_in = sampler1.get_samples_alt(prior_obj_name="hidden_in",permuted=False)
mcmc_samples_hidden_out = sampler1.get_samples_alt(prior_obj_name="hidden_out",permuted=False)

#print(mcmc_samples_beta["indices_dict"])
#exit()

samples = mcmc_samples_hidden_in["samples"]
hidden_in_tau_indices = mcmc_samples_hidden_in["indices_dict"]["tau"]
hidden_in_w_indices = mcmc_samples_hidden_in["indices_dict"]["w"]
hidden_out_tau_indices = mcmc_samples_hidden_out["indices_dict"]["tau"]
hidden_out_w_indices = mcmc_samples_hidden_out["indices_dict"]["w"]
#print(samples.shape)
posterior_mean_hidden_in_tau = numpy.mean(samples[:,:,hidden_in_tau_indices].reshape(-1,len(hidden_in_tau_indices)),axis=0)


print("hidden in tau {}".format(posterior_mean_hidden_in_tau))

#print(mcmc_samples_beta["indices_dict"])

out = sampler1.get_diagnostics(permuted=False)


#processed_diag = process_diagnostics(out,name_list=["accepted"])
#print(processed_diag.shape)

#processed_energy = process_diagnostics(out,name_list=["prop_H"])

print(energy_diagnostics(diagnostics_obj=out))