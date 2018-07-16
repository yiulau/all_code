import numpy
from experiments.experiment_util import wishart_for_cov
from distributions.mvn import V_mvn
from abstract.util import wrap_V_class_with_input_data
from experiments.float_vs_double.convergence.util import convert_convergence_output_to_numpy
from experiments.float_vs_double.convergence.float_vs_double_convergence import convergence_diagnostics
from input_data.convert_data_to_dict import get_data_dict
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from abstract.util import wrap_V_class_with_input_data

#####################################################################################################################################
numpy.random.seed(1)
input_data = {"input":wishart_for_cov(dim=10)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
V_mvn2 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
####################################################################################################################################
# logisitc regressions
input_data_pima_indian = get_data_dict("pima_indian")
V_pima_indian = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_pima_indian)


input_data_australian = get_data_dict("australian")
V_australian = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_australian)

input_data_heart = get_data_dict("heart")
V_heart = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_heart)

input_data_breast = get_data_dict("breast")
V_breast = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_breast)

input_data_german = get_data_dict("german")
V_german = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data_german)

#####################################################################################################################################
v_fun_list = [V_mvn1,V_pima_indian,V_australian,V_heart,V_breast,V_german]
#model_names = ["mvn","pima_indian","australian","heart","breast","german"]
v_fun_list = [V_mvn1,V_pima_indian]
model_names = ["mvn","pima_indian"]
#out = convergence_diagnostics(v_fun=V_mvn1,seed=18)
num_models = len(v_fun_list)
store_results = numpy.zeros((num_models,6))
#print(out)
for i in range(len(v_fun_list)):
    out = convergence_diagnostics(v_fun=v_fun_list[i],seed=i+1)
    store_results[i,:] = convert_convergence_output_to_numpy(out)


to_store = {"results":store_results,"model_names":model_names}
print(store_results)

numpy.savez("float_v_double_convergence.npz",**to_store)


