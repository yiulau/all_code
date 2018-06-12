from post_processing.diagnostics import WAIC
import pickle
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from input_data.convert_data_to_dict import get_data_dict

with open("debug_test_error_mcmc.pkl", 'rb') as f:
    mcmc_samples = pickle.load(f)


train_data = get_data_dict("pima_indian")

out_waic = WAIC(mcmc_samples,train_data,V_pima_inidan_logit())

