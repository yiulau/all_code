import numpy,torch
from experiments.experiment_util import wishart_for_cov
from distributions.mvn import V_mvn
from abstract.util import wrap_V_class_with_input_data
from experiments.float_vs_double.stability.leapfrog_stability import generate_q_list,generate_Hams,leapfrog_stability_test
from abstract.abstract_class_point import point
from experiments.float_vs_double.convergence.float_vs_double_convergence import convergence_diagnostics
numpy.random.seed(1)
input_data = {"input":wishart_for_cov(dim=10)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)

out = generate_q_list(v_fun=V_mvn1,num_of_pts=3)

list_q_double = out["list_q_double"]
list_q_float = out["list_q_float"]

#print(len(list_q_double))
#print(len(list_q_float))

print(list_q_double[0].flattened_tensor)
# #
#
print(list_q_double[2].flattened_tensor)
# #
print(list_q_float[0].flattened_tensor)
# #
print(list_q_float[2].flattened_tensor)

list_p_double = [None]*len(list_q_double)
list_p_float = [None]*len(list_q_float)

for i in range(len(list_q_float)):
    p_double = list_q_double[i].point_clone()
    momentum = torch.randn(len(p_double.flattened_tensor)).type("torch.DoubleTensor")
    p_double.flattened_tensor.copy_(momentum)
    p_double.load_flatten()
    p_float = list_q_float[i].point_clone()
    p_float.flattened_tensor.copy_(momentum.type("torch.FloatTensor"))
    p_float.load_flatten()

    list_p_double[i] = p_double
    list_p_float[i] = p_float


out = generate_Hams(v_fun=V_mvn1)

Ham_float = out["float"]
print(Ham_float.V.flattened_tensor)
Ham_double = out["double"]

# q = point(V=V_mvn1(precision_type="torch.FloatTensor"))
# p = q.point_clone()
# q.flattened_tensor.normal_()
# p.flattened_tensor.normal_()
# q.load_flatten()
# p.load_flatten()

#import dill as pickle

# out = {"Ham_float":Ham_float,"list_q_float":list_q_float,"list_p_float":list_p_float}
# with open("debug_leapfrog_stability.pkl", 'wb') as f:
#     pickle.dump(out, f)


out_double = leapfrog_stability_test(Ham=Ham_double,epsilon=0.01,L=1000,list_q=list_q_double,list_p=list_p_double,precision_type="torch.DoubleTensor")
print(out_double)

out_float = leapfrog_stability_test(Ham=Ham_float,epsilon=0.01,L=1000,list_q=list_q_float,list_p=list_p_float,precision_type="torch.FloatTensor")
print(out_float)

