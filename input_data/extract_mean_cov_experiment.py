from input_data.extract_logit_mean_cov import compute_store_logit_mean_cov
from input_data.convert_data_to_dict import get_data_dict

#target = get_data_dict("logistic_mnist")["target"]
#X = get_data_dict("logistic_mnist",standardize_predictor=False)["input"]
#print(X[0,:])
#print(target)
#exit()


data_name_list = ["australian","german","heart","breast"]

for name in data_name_list:
    compute_store_logit_mean_cov(name,True)
    compute_store_logit_mean_cov(name,False)


