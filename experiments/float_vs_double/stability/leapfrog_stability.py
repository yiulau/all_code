from abstract.abstract_leapfrog_util import abstract_leapfrog_ult
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from abstract.abstract_class_point import point
import numpy
def leapfrog_stability_test(Ham,epsilon,L,list_q,list_p):
    out = [None]*len(list_q)
    for i in range(len(list_q)):
        Ham.V.load_point(list_q[i])
        Ham.T.load_point(list_p[i])
        begin_H = Ham.evaluate()
        q = list_q[i]
        p = list_p[i]
        for cur in range(L):
            out_q,out_p,stat = abstract_leapfrog_ult(q,p,epsilon,Ham)
            if stat["divergent"]:
                out[i]="divergent"
                break
            else:
                q = out_q
                p = out_p

        if not out[i]=="divergent":
            end_H = Ham.evaluate(q,p)
            out[i] = (end_H-begin_H)

    return(out)

def generate_Hams(v_fun):
    out = {"float":None,"double":None}
    v_obj_float = v_fun(precision_type="torch.FloatTensor")
    v_obj_double = v_fun(precision_type="torch.DoubleTensor")
    metric_float = metric("unit_e",v_obj_float)
    metric_double = metric("unit_e",v_obj_double)

    Ham_float = Hamiltonian(v_obj_float,metric_float)
    Ham_double = Hamiltonian(v_obj_double,metric_double)

    out.update({"float":Ham_float,"double":Ham_double})
    return(out)


def generate_q_p_list(v_fun,num_of_pts):

    mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=10000, num_chains=4, num_cpu=1, thin=1,
                                           tune_l_per_chain=1000,
                                           warmup_per_chain=1100, is_float=False, isstore_to_disk=False,allow_restart=True)
    input_dict = {"v_fun": [v_fun], "epsilon": ["dual"], "second_order": [False], "cov": ["adapt"],
                  "metric_name": ["dense_e"], "dynamic": [False], "windowed": [False],
                  "criterion": ["gnuts"]}

    ep_dual_metadata_argument = {"name": "epsilon", "target": 0.65, "gamma": 0.05, "t_0": 10,
                                 "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}

    dim = len(v_fun().flattened_tensor)
    adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=dim)]
    dual_args_list = [ep_dual_metadata_argument]
    other_arguments = other_default_arguments()

    tune_settings_dict = tuning_settings(dual_args_list, [], adapt_cov_arguments, other_arguments)

    tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

    sampler1 = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta, tune_settings_dict=tune_settings_dict)

    out = sampler1.start_sampling()
    sampler1.remove_failed_chains()

    print("num chains removed {}".format(sampler1.metadata.num_chains_removed))
    print("num restarts {}".format(sampler1.metadata.num_restarts))

    samples = sampler1.get_samples_p_diag(permuted=True)

    num_mcmc_samples = samples.shape[0]
    indices = numpy.random.choice(a=num_mcmc_samples,size=num_of_pts,replace=False)

    list_q_double = [None]*num_of_pts
    list_p_double = [None]*num_of_pts

    for i in range(num_of_pts):
        q_point = point(V=v_fun())
        p_point = point(list_tensor=q_point.list_tensor,pointtype="p",need_flatten=q_point.need_flatten)
        p_point.flattened_tensor.normal_()
        p_point.load_flatten()
        list_q_double[i] = q_point
        list_p_double[i] = p_point

    list_q_float = [None] * num_of_pts
    list_p_float = [None] * num_of_pts

    for i in range(num_of_pts):
        list_q_float[i] = list_q_double[i].clone_cast_type("torch.FloatTensor")
        list_p_float[i] = list_p_double[i].clone_cast_type("torch.FloatTensor")


    out = {"list_q_double":list_q_double,"list_p_double":list_p_double,"list_q_float":list_q_float,"list_p_float":list_p_float}
    return(out)












