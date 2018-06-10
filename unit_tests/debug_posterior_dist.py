from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import map_prediction,test_error,posterior_predictive_dist
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
import pickle, torch
with open("debug_test_error_mcmc.pkl", 'rb') as f:
    mcmc_samples = pickle.load(f)

target_dataset = get_data_dict("pima_indian")

out_dist = posterior_predictive_dist(target_dataset,V_pima_inidan_logit(),mcmc_samples,"classification")

print(out_dist.shape)

out_dist2 = torch.zeros(out_dist.shape)
v_nn_obj = V_pima_inidan_logit()
for i in range(out_dist.shape[0]):
    test_samples = target_dataset["input"][i:i+1,:]
    for j in range(out_dist.shape[2]):
        v_nn_obj.flattened_tensor.copy_(torch.from_numpy(mcmc_samples[j, :]))
        v_nn_obj.load_flattened_tensor_to_param()
        out_prob = v_nn_obj.predict(test_samples)
        out_dist2[i,:,j] = out_prob

diff_out_dist = ((out_dist-out_dist2)*(out_dist-out_dist2)).sum()

print("diff dist {}".format(diff_out_dist))