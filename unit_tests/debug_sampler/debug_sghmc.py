from experiments.neural_net_experiments.sghmc_vs_batch_hmc.sghmc_batchhmc import sghmc_sampler
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from abstract.metric import metric
from input_data.convert_data_to_dict import get_data_dict
import numpy
from abstract.abstract_class_point import point
from abstract.abstract_class_Ham import Hamiltonian
v_obj = V_pima_inidan_logit(precision_type="torch.DoubleTensor")
metric = metric(name="unit_e",V_instance=v_obj)
Ham = Hamiltonian(V=v_obj,metric=metric)

full_data = get_data_dict("pima_indian")
init_q_point = point(V=v_obj)
out = sghmc_sampler(init_q_point=init_q_point,epsilon=0.01,L=10,Ham=Ham,alpha=0.01,eta=0.01,
              betahat=0,full_data=full_data,num_samples=1000,thin=0,burn_in=200,batch_size=50)

store = out[0]

print(store.shape)

print(numpy.mean(store.numpy(),axis=0))





