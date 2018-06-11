from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import map_prediction,test_error
from distributions.linear_regressions.linear_regression import V_linear_regression
import pickle
with open("debug_test_error_mcmc_regression.pkl", 'rb') as f:
    mcmc_samples = pickle.load(f)

#print(mcmc_samples.shape)

target_dataset = get_data_dict("boston")



te1,predicted1 = test_error(target_dataset,v_obj=V_linear_regression(),mcmc_samples=mcmc_samples,type="regression",memory_efficient=False)
te2,predicted2 = test_error(target_dataset,v_obj=V_linear_regression(),mcmc_samples=mcmc_samples,type="regression",memory_efficient=True)

print(te1)
print(te2)

#print(sum(predicted1!=predicted2))