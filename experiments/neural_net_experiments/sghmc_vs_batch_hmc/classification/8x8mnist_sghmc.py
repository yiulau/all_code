from experiments.neural_net_experiments.sghmc_vs_batch_hmc.sghmc_batchhmc import sghmc_sampler
from experiments.neural_net_experiments.sghmc_vs_batch_hmc.model import V_fc_model_1
from abstract.metric import metric
from input_data.convert_data_to_dict import get_data_dict
import numpy
from abstract.abstract_class_point import point
from abstract.abstract_class_Ham import Hamiltonian
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error

input_data = get_data_dict("8x8mnist",standardize_predictor=True)
test_set = {"input":input_data["input"][-500:,],"target":input_data["target"][-500:]}
train_set = {"input":input_data["input"][:500,],"target":input_data["target"][:500]}



prior_dict = {"name":"normal"}
model_dict = {"num_units":25}

v_generator =wrap_V_class_with_input_data(class_constructor=V_fc_model_1,input_data=train_set,prior_dict=prior_dict,model_dict=model_dict)

v_obj = v_generator(precision_type="torch.DoubleTensor")
metric = metric(name="unit_e",V_instance=v_obj)
Ham = Hamiltonian(V=v_obj,metric=metric)

full_data = input_data
init_q_point = point(V=v_obj)
out = sghmc_sampler(init_q_point=init_q_point,epsilon=1e-4,L=2,Ham=Ham,alpha=0.01,eta=1e-5,
              betahat=0,full_data=full_data,num_samples=10000,thin=0,burn_in=200,batch_size=25)

store = out[0]

print(store.shape)

print(numpy.mean(store.numpy(),axis=0))

v_generator = wrap_V_class_with_input_data(class_constructor=V_fc_model_1,input_data=input_data,prior_dict=prior_dict,model_dict=model_dict)
precision_type = "torch.DoubleTensor"
#test_mcmc_samples = numpy.zeros((1,store.shape[1]))
test_mcmc_samples = store.numpy()

te1,predicted1 = test_error(test_set,v_obj=v_generator(precision_type=precision_type),mcmc_samples=test_mcmc_samples,type="classification",memory_efficient=False)

print(te1)