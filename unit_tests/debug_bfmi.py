from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import map_prediction,test_error
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from post_processing.ESS_nuts import diagnostics_stan
from post_processing.get_diagnostics import get_diagnostics_from_sample
import pickle
with open("debug_test_error_mcmc.pkl", 'rb') as f:
    out = pickle.load(f)


mcmc_samples = out["samples"]
diagnostics = out["diagnostics"]


H_diagnostics = get_diagnostics_from_sample(diagnostics,permuted=False,name="prop_H")

print(H_diagnostics.shape)

diagnostics_stan(H_diagnostics)
