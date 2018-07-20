import numpy
import torch
import torch.nn as nn
from abstract.abstract_class_V import V
from torch.autograd import Variable
from general_util.pytorch_random import generate_gamma
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_class_point import point
from explicit.general_util import logsumexp_torch
from experiments.neural_net_experiments.gibbs_vs_joint_sampling.gibbs_vs_together_hyperparam import update_param_and_hyperparam_one_step,update_param_and_hyperparam_dynamic_one_step
from abstract.mcmc_sampler import log_class
from input_data.convert_data_to_dict import get_data_dict
from post_processing.test_error import test_error
from abstract.abstract_nuts_util import abstract_GNUTS
from general_util.pytorch_random import log_inv_gamma_density
from post_processing.ESS_nuts import diagnostics_stan
from experiments.neural_net_experiments.gibbs_vs_joint_sampling.V_hierarchical_fc1 import V_fc_gibbs_model_1
from abstract.util import wrap_V_class_with_input_data
# compare sampling with cp and ncp (joint sampling)
# compare gibbs (cp) and joint sampling (cp, ncp)
# unit_e
# point is to show that sampling can be successful , i.e. reasonably large ess. no divergence
# gaussian inv gamma prior

# compare ess for hyperparameter

input_data = get_data_dict("8x8mnist")
input_data = {"input":input_data["input"][:500,],"target":input_data["target"][:500]}
model_dict = {"num_units":25}
V_fun = wrap_V_class_with_input_data(class_constructor=V_fc_gibbs_model_1,input_data=input_data,model_dict=model_dict)
v_obj = V_fun(precision_type="torch.DoubleTensor",gibbs=True)
metric_obj = metric(name="unit_e",V_instance=v_obj)
Ham = Hamiltonian(v_obj,metric_obj)

init_q_point = point(V=v_obj)
init_hyperparam = torch.abs(torch.randn(1))+3
log_obj = log_class()

#print(init_q_point.flattened_tensor)

num_samples = 1000
dim = len(init_q_point.flattened_tensor)
mcmc_samples_weight = torch.zeros(1,num_samples,dim)
mcmc_samples_hyper = torch.zeros(1,num_samples,1)
for i in range(num_samples):
    print("loop {}".format(i))
    #outq,out_hyperparam = update_param_and_hyperparam_one_step(init_q_point,init_hyperparam,Ham,0.1,60,log_obj)
    outq, out_hyperparam = update_param_and_hyperparam_dynamic_one_step(init_q_point, init_hyperparam, Ham, 0.01, log_obj)
    init_q_point.flattened_tensor.copy_(outq.flattened_tensor)
    init_q_point.load_flatten()
    init_hyperparam = out_hyperparam
    print("sigma2 {}".format(init_hyperparam))
    mcmc_samples_weight[0,i,:] = outq.flattened_tensor.clone()
    mcmc_samples_hyper[0,i,0] = out_hyperparam
mcmc_samples_weight = mcmc_samples_weight.numpy()
mcmc_samples_hyper = mcmc_samples_hyper.numpy()
print(mcmc_samples_weight.shape)

print("sigma2 diagnostics gibbs")
print(mcmc_samples_hyper)
print(numpy.mean(mcmc_samples_hyper))
print(numpy.var(mcmc_samples_hyper))
print(diagnostics_stan(mcmc_samples_tensor=mcmc_samples_hyper))

print("weight diagnostics gibbs")
# print(numpy.mean(mcmc_samples_weight,axis=0))
print(min(diagnostics_stan(mcmc_samples_tensor=mcmc_samples_weight)["ess"]))


test_mcmc_samples = numpy.zeros((1,mcmc_samples_weight.shape[2]))
test_mcmc_samples = mcmc_samples_weight[0,:,:]

te2,predicted2 = test_error(input_data,v_obj=V_fun(precision_type="torch.DoubleTensor"),mcmc_samples=test_mcmc_samples,type="classification",memory_efficient=False)

print(te2)
