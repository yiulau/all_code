from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import test_error
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from abstract.util import wrap_V_class_with_input_data
import pickle

with open("debug_test_error_mcmc.pkl", 'rb') as f:
     sampler1 = pickle.load(f)

input_data = get_data_dict("pima_indian")
v_generator = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)
mcmc_samples_mixed = sampler1.get_samples(permuted=True)
precision_type = "torch.DoubleTensor"
te1,predicted1 = test_error(input_data,v_obj=v_generator(precision_type=precision_type),mcmc_samples=mcmc_samples_mixed,type="classification",memory_efficient=False)
te2,predicted2 = test_error(input_data,v_obj=v_generator(precision_type=precision_type),mcmc_samples=mcmc_samples_mixed,type="classification",memory_efficient=True)

print(te1)
print(te2)

print(sum(predicted1!=predicted2))