from post_processing.test_error import test_error
from abstract.abstract_class_point import point
from input_data.convert_data_to_dict import get_data_dict
from abstract.util import wrap_V_class_with_input_data
from unit_tests.debug_optimization.debug_nn_raw.model import V_fc_model_1
from unit_tests.debug_optimization.debug_optimization import gradient_descent
import torch

input_data = get_data_dict("mnist") # 0.866 when num_units = 10
input_data = {"input":input_data["input"][:500,:],"target":input_data["target"][:500]}
#input_data = get_data_dict("mnist")
prior_dict = {"name":"normal"}
model_dict = {"num_units":150}

v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_model_1,input_data=input_data,prior_dict=prior_dict,model_dict=model_dict)

out,explode_grad = gradient_descent(number_of_iter=1000,lr=0.001,v_obj=v_generator(precision_type="torch.DoubleTensor"))

#print(out.flattened_tensor)

mcmc_samples = torch.zeros(1,len(out.flattened_tensor))
mcmc_samples[0,:] = out.flattened_tensor
mcmc_samples = mcmc_samples.numpy()



te,predicted = test_error(target_dataset=input_data,v_obj=v_generator(precision_type="torch.DoubleTensor"),mcmc_samples=mcmc_samples,type="classification")

print(te)

