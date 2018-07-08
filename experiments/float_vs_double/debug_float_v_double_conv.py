import numpy
from experiments.experiment_util import wishart_for_cov
from distributions.mvn import V_mvn
from abstract.util import wrap_V_class_with_input_data

from experiments.float_vs_double.convergence.float_vs_double_convergence import convergence_diagnostics
numpy.random.seed(1)
input_data = {"input":wishart_for_cov(dim=10)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
V_mvn2 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)
v_fun_list = [V_mvn1,V_mvn2]

out = convergence_diagnostics(v_fun=V_mvn1,seed=18)
print(input_data["input"][1,1])
print(out)