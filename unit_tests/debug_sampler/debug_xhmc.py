from abstract.abstract_nuts_util import abstract_NUTS_xhmc
from abstract.abstract_class_Ham import Hamiltonian
from abstract.metric import metric
from abstract.abstract_class_point import point
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
import torch,numpy,os,pickle
from experiments.correctdist_experiments.prototype import check_mean_var_stan
from abstract.util import wrap_V_class_with_input_data
from input_data.convert_data_to_dict import get_data_dict
input_data = get_data_dict("pima_indian")
V_pima_indian_logit = wrap_V_class_with_input_data(class_constructor=V_logistic_regression,input_data=input_data)
v_obj = V_pima_indian_logit(precision_type="torch.DoubleTensor")
metric_obj = metric("unit_e",v_obj)
Ham = Hamiltonian(v_obj,metric_obj)

q_point = point(V=Ham.V)
inputq = torch.randn(len(q_point.flattened_tensor))
q_point.flattened_tensor.copy_(inputq)
q_point.load_flatten()

chain_l=200
store_samples = torch.zeros(1,chain_l,len(inputq))
for i in range(chain_l):
    out = abstract_NUTS_xhmc(init_q=q_point,epsilon=0.1,Ham=Ham,xhmc_delta=0.1,max_tree_depth=10)
    store_samples[0,i,:] = out[0].flattened_tensor.clone()
    q_point = out[0]


store = store_samples.numpy()
#empCov = numpy.cov(store,rowvar=False)
#emmean = numpy.mean(store,axis=0)

mcmc_samples = store
#print(emmean)
#print(empCov)

address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain.pkl"
correct = pickle.load(open(address, 'rb'))
correct_mean = correct["correct_mean"]
#print("correct_mean {}".format(correct_mean))
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()
#print("correct_cov {}".format(correct_cov))

output = check_mean_var_stan(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = output["mcmc_mean"],output["mcmc_Cov"]
pc_mean,pc_cov = output["pc_of_mean"],output["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)