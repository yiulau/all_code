from post_processing.diagnostics import WAIC,convert_mcmc_tensor_to_list_points,psis
import pickle,torch
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from input_data.convert_data_to_dict import get_data_dict


with open("debug_test_error_mcmc.pkl", 'rb') as f:
    sampler = pickle.load(f)

#print(mcmc_samples["samples"].shape)
#print(mcmc_samples["samples"])
#exit()
train_data = get_data_dict("pima_indian")

v_obj = V_pima_inidan_logit(precision_type="torch.DoubleTensor")

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
mcmc_tensor = torch.from_numpy(sampler.get_samples(permuted=True))
chains_combined_mcmc_tensor = mcmc_tensor
list_mcmc_point = convert_mcmc_tensor_to_list_points(chains_combined_mcmc_tensor,v_obj)

out_waic = WAIC(list_mcmc_point,train_data,v_obj)

print(out_waic)
