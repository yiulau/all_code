import dill as pickle
from experiments.float_vs_double.stability.leapfrog_stability import generate_q_list,generate_Hams,leapfrog_stability_test
out = pickle.load(open("debgu_leapfrog_stability.pkl", 'rb'))


Ham_float = out["Ham_float"]
list_q_float = out["list_q_float"]
list_p_float = out["list_p_float"]

out_float = leapfrog_stability_test(Ham=Ham_float,epsilon=0.01,L=1000,list_q=list_q_float,list_p=list_p_float)
print(out_float)
