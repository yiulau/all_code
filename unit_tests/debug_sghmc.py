from experiments.neural_net_experiments.sghmc_batchhmc import sghmc_sampler
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from abstract.metric import metric
from input_data.convert_data_to_dict import get_data_dict
from abstract.abstract_class_point import point
from abstract.abstract_class_Ham import Hamiltonian
v_obj = V_pima_inidan_logit()
metric = metric(name="unit_e",V_instance=v_obj)
Ham = Hamiltonian(V=v_obj,metric=metric)

full_data = get_data_dict("pima_indian")
init_q_point = point(V=v_obj)
sghmc_sampler(init_q_point=init_q_point,epsilon=0.1,L=10,Ham=Ham,alpha=0.01,eta=0.1*1e-5,
              betahat=0,full_data=full_data,num_samples=1000,thin=0,burn_in=200)

