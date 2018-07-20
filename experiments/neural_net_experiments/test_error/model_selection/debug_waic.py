from post_processing.diagnostics import WAIC,convert_mcmc_tensor_to_list_points
import pickle,torch
from input_data.convert_data_to_dict import get_data_dict
from distributions.neural_nets.fc_V_model_4 import V_fc_model_4
from abstract.util import wrap_V_class_with_input_data

with open("55_debug_waic_fc1_sampler.pkl", 'rb') as f:
    sampler = pickle.load(f)


samples = sampler.get_samples(permuted=True)

#print(mcmc_samples["samples"].shape)
#print(mcmc_samples["samples"])
#exit()
#train_data = get_data_dict("8x8mnist")
train_data = get_data_dict("8x8mnist",standardize_predictor=True)

train_data = {"input":train_data["input"][:500,:],"target":train_data["target"][:500]}

prior_dict = {"name":"normal"}
model_dict = {"num_units":55}
v_fun = wrap_V_class_with_input_data(class_constructor=V_fc_model_4,input_data=train_data,prior_dict=prior_dict,model_dict=model_dict)
v_obj = v_fun(precision_type="torch.DoubleTensor")

#print(v_obj.beta)
#exit()
# out0 = v_obj.forward()
#
# out_sum = 0
# for i in range(len(train_data["input"])):
#     out_sum += v_obj.forward(input={"input":torch.from_numpy(train_data["input"][i:i+1,:]),"target":torch.from_numpy(train_data["target"][i:i+1])}).data[0]
#
# print(out_sum)
# print(out0)
# exit()

#mcmc_tensor = torch.from_numpy(mcmc_samples["samples"])
#print(mcmc_tensor.shape)
#print(mcmc_tensor.view(-1,7).shape)
#exit()
chains_combined_mcmc_tensor = torch.from_numpy(samples)
list_mcmc_point = convert_mcmc_tensor_to_list_points(chains_combined_mcmc_tensor,v_obj)

out_waic = WAIC(list_mcmc_point,train_data,v_obj)

print(out_waic)

# need to specify likelihood including the normalizing constant