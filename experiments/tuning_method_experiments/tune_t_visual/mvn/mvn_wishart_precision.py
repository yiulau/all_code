import numpy,torch
import dill as pickle
from experiments.experiment_util import wishart_for_cov
from abstract.util import wrap_V_class_with_input_data
from distributions.mvn import V_mvn
from abstract.abstract_class_point import point
from experiments.tuning_method_experiments.tune_t_visual.util import generate_H_V_T
from experiments.float_vs_double.stability.leapfrog_stability import generate_q_p_list

from post_processing.plot_energy_oscillation import plot_V_T,plot_V_H

input_data = {"input":wishart_for_cov(dim=50)}
V_mvn1 = wrap_V_class_with_input_data(class_constructor=V_mvn,input_data=input_data)


# return list of q's generate from adaptive sampling

sampler_address = ".pkl"

sampler1 = pickle.load(open(sampler_address, 'rb'))

samples = sampler1.get_samples(permuted=True)

num_samples = samples.shape[0]
num_chosen = 3
seed = 12
numpy.random.seed(seed)
indices = numpy.random.uniform(0,num_samples,num_chosen)
indices = indices.astype(int)
indices = [numpy.asscalar(y) for y in indices]

q_point_list = [None]*list
for i in range(len(indices)):
    flattened_tensor = samples[:,indices[i]]
    flattened_tensor = torch.from_numpy(flattened_tensor)
    q_point = point(V=V_mvn1(precision_type="torch.DoubleTensor"))
    q_point.flattened_tensor.copy_(flattened_tensor)
    q_point.load_flatten()
    q_point_list[i] = q_point
#############################################################################################################################
# q_point = point(V=V_mvn1(precision_type="torch.DoubleTensor"))
#
# q_point.flattened_tensor.normal_()
# q_point.load_flatten()
# save_name = "pic.png"
# out = generate_H_V_T(epsilon=0.001,L=11000,vo=V_mvn1(precision_type="torch.DoubleTensor"),q_point=q_point)
#
# V_vec = numpy.array(out["V_list"])
# T_vec = numpy.array(out["T_list"])
# # print(len(V_vec))
# # print(len(T_vec))
#
#
# #plot_V_T(V_vec=V_vec[-2000:],T_vec=T_vec[-2000:],epsilon=0.01)
#
# plot_V_H(V_vec=V_vec[-10000:],T_vec=T_vec[-10000:],epsilon=0.001,save_name="{}_".format(0)+save_name)
#exit()
for i in range(len(q_point_list)):
    out = generate_H_V_T(epsilon=0.0001,L=110000,vo=V_mvn1(precision_type="torch.DoubleTensor"),q_point=q_point)

    V_vec = numpy.array(out["V_list"])
    T_vec = numpy.array(out["T_list"])
    # print(len(V_vec))
    # print(len(T_vec))


    #plot_V_T(V_vec=V_vec[-2000:],T_vec=T_vec[-2000:],epsilon=0.01)

    plot_V_H(V_vec=V_vec[-100000:],T_vec=T_vec[-100000:],epsilon=0.0001,save_name="{}_".format(i)+save_name)









