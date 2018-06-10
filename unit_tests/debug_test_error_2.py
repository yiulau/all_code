from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import map_prediction,test_error
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
import pickle
with open("debug_test_error_mcmc.pkl", 'rb') as f:
    mcmc_samples = pickle.load(f)

target_dataset = get_data_dict("pima_indian")



te1,predicted1 = test_error(target_dataset,v_obj=V_pima_inidan_logit(),mcmc_samples=mcmc_samples,type="classification",memory_efficient=False)
te2,predicted2 = test_error(target_dataset,v_obj=V_pima_inidan_logit(),mcmc_samples=mcmc_samples,type="classification",memory_efficient=True)

print(te1)
print(te2)

print(sum(predicted1!=predicted2))