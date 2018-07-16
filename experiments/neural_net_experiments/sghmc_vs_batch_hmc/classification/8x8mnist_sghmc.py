from experiments.neural_net_experiments.sghmc_vs_batch_hmc.sghmc_batchhmc import sghmc_sampler
from experiments.neural_net_experiments.sghmc_vs_batch_hmc.model import V_fc_model_1
from abstract.metric import metric
from input_data.convert_data_to_dict import get_data_dict
import numpy
from abstract.abstract_class_point import point
from abstract.abstract_class_Ham import Hamiltonian
from abstract.util import wrap_V_class_with_input_data

input_data = get_data_dict("8x8mnist",standardize_predictor=True)


prior_dict = {"name":"gaussian_inv_gamma_2"}
model_dict = {"num_units":25}

v_generator =wrap_V_class_with_input_data(class_constructor=V_fc_model_1,input_data=input_data,prior_dict=prior_dict,model_dict=model_dict)

v_obj = v_generator(precision_type="torch.DoubleTensor")
metric = metric(name="unit_e",V_instance=v_obj)
Ham = Hamiltonian(V=v_obj,metric=metric)

full_data = get_data_dict("8x8mnist")
init_q_point = point(V=v_obj)
out = sghmc_sampler(init_q_point=init_q_point,epsilon=0.2,L=20,Ham=Ham,alpha=0.01,eta=0.01,
              betahat=0,full_data=full_data,num_samples=1000,thin=0,burn_in=200,batch_size=50)

store = out[0]

print(store.shape)

print(numpy.mean(store.numpy(),axis=0))
