import numpy,os

from distributions.mvn import V_mvn
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict
from experiments.float_vs_double.convergence.match_mean_cov_convergence.util import match_convergence_test
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from abstract.util import wrap_V_class_with_input_data
from experiments.experiment_util import wishart_for_cov
correct_mean_list = []
correct_cov_list = []
v_fun_list = []
#####################################################################################################################################
input_data = {"input":wishart_for_cov(dim=100,seed=100)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
correct_mean_list.append(numpy.zeros(100))
correct_cov_list.append(numpy.linalg.inv(input_data["input"]))
v_fun_list.append(V_mvn1)
####################################################################################################################################
# logisitc regressions
input_data_pima_indian = get_data_dict("pima_indian")
V_pima_indian = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_pima_indian)
v_fun_list.append(V_pima_indian)

input_data_australian = get_data_dict("australian")
V_australian = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_australian)
v_fun_list.append(V_australian)

input_data_heart = get_data_dict("heart")
V_heart = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_heart)
v_fun_list.append(V_heart)

input_data_breast = get_data_dict("breast")
V_breast = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_breast)
v_fun_list.append(V_breast)

input_data_german = get_data_dict("german")
V_german = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_german)
v_fun_list.append(V_german)

names_list =["pima_indian","australian","heart","breast","german"]

for i in range(len(names_list)):
    address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain/logistic/result_from_long_chain_{}.npz".format(names_list[i])
    result = numpy.load(address)
    correct_mean = result["correct_mean"]
    correct_cov = result["correct_cov"]
    correct_mean_list.append(correct_mean)
    correct_cov_list.append(correct_cov)


names_list = ["mvn"]+names_list
for i in range(len(names_list)):
    out = match_convergence_test(v_fun=v_fun_list[i],seed=i+1,correct_mean=correct_mean_list[i],correct_cov=correct_cov_list[i])
    save_name = "match_convergence_result_{}.npz".format(names_list[i])
    numpy.savez(save_name,**out)



