from post_processing.test_error import test_error
from abstract.abstract_class_point import point
from input_data.convert_data_to_dict import get_data_dict
from abstract.util import wrap_V_class_with_input_data
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from unit_tests.debug_optimization.debug_optimization import gradient_descent
import torch

input_data = get_data_dict("pima_indian")

v_generator = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)

out,explode_grad = gradient_descent(number_of_iter=100,lr=0.01,v_obj=v_generator(precision_type="torch.DoubleTensor"))

print(out.flattened_tensor)

mcmc_samples = torch.zeros(1,len(out.flattened_tensor))
mcmc_samples[0,:] = out.flattened_tensor
mcmc_samples = mcmc_samples.numpy()
te,predicted = test_error(target_dataset=input_data,v_obj=v_generator(precision_type="torch.DoubleTensor"),mcmc_samples=mcmc_samples,type="classification")

print(te)







