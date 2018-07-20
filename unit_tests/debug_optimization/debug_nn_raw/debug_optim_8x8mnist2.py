from post_processing.test_error import test_error
from abstract.abstract_class_point import point
from input_data.convert_data_to_dict import get_data_dict
from abstract.util import wrap_V_class_with_input_data
from distributions.neural_nets.fc_V_model_debug import V_fc_model_debug
from unit_tests.debug_optimization.debug_optimization import gradient_descent
import torch

input_data = get_data_dict("8x8mnist") # 0.866 when num_units = 10
test_set = {"input":input_data["input"][:-500,],"target":input_data["target"][:-500]}
input_data = {"input":input_data["input"][:500,],"target":input_data["target"][:500]}

model_dict = {"num_units":50}

v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_model_debug,input_data=input_data,model_dict=model_dict)

out,explode_grad = gradient_descent(number_of_iter=2000,lr=0.01,v_obj=v_generator(precision_type="torch.DoubleTensor"))

#print(out.flattened_tensor)

mcmc_samples = torch.zeros(1,len(out.flattened_tensor))
mcmc_samples[0,:] = out.flattened_tensor
print(max(torch.abs(out.flattened_tensor)))
print(min(torch.abs(out.flattened_tensor)))
mcmc_samples = mcmc_samples.numpy()

v_obj = v_generator(precision_type="torch.DoubleTensor")
v_obj.flattened_tensor.copy_(out.flattened_tensor)
v_obj.load_flattened_tensor_to_param()
print(v_obj.forward())

#print((v_obj.flattened_tensor*v_obj.flattened_tensor).sum())
out = v_obj.predict(inputX=input_data["input"])
prediction = torch.max(out,1)[1]
#print(out.shape)
#print(out)
print(prediction[60:80].numpy())
print(input_data["target"][60:80])
te,predicted = test_error(target_dataset=test_set,v_obj=v_generator(precision_type="torch.DoubleTensor"),mcmc_samples=mcmc_samples,type="classification")

print(te)