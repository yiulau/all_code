from abstract.abstract_leapfrog_util import abstract_leapfrog_ult
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.mcmc_sampler import mcmc_sampler, mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from experiments.experiment_obj import tuneinput_class
from abstract.abstract_class_point import point
import numpy,torch
# assume unit_e metric
def leapfrog_stability_test(Ham,epsilon,L,list_q,list_p,precision_type):

    torch.set_default_tensor_type(precision_type)
    out = [None]*len(list_q)
    for i in range(len(list_q)):
        #print(q)

        q = list_q[i]
        p = list_p[i]
        Ham.V.load_point(q)
        Ham.T.load_point(p)
        begin_H = Ham.evaluate(q,p)["H"]
        print(begin_H)
        print(q.flattened_tensor)
        print(p.flattened_tensor)

        for cur in range(L):
            out_q,out_p,stat = abstract_leapfrog_ult(q,p,epsilon,Ham)
            print(out_q.flattened_tensor)
            if stat["explode_grad"]:
                out[i]="divergent"
                break
            else:
                q = out_q
                p = out_p

        if not out[i]=="divergent":
            end_H = Ham.evaluate(q,p)["H"]
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


def generate_q_list(v_fun,num_of_pts):
    # extract number of (q,p) points given v_fun
    mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0, samples_per_chain=2000, num_chains=4, num_cpu=4, thin=1,
                                           tune_l_per_chain=900,
                                           warmup_per_chain=1000, is_float=False, isstore_to_disk=False,allow_restart=True)
    input_dict = {"v_fun": [v_fun], "epsilon": ["dual"], "second_order": [False],"cov":["adapt"],
                  "metric_name": ["diag_e"], "dynamic": [True], "windowed": [False],"max_tree_depth":[8],
                  "criterion": ["xhmc"],"xhmc_delta":[0.1]}

    ep_dual_metadata_argument = {"name": "epsilon", "target": 0.9, "gamma": 0.05, "t_0": 10,
                                 "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}

    dim = len(v_fun(precision_type="torch.DoubleTensor").flattened_tensor)
    adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=dim)]
    dual_args_list = [ep_dual_metadata_argument]
    other_arguments = other_default_arguments()

    tune_settings_dict = tuning_settings(dual_args_list, [],adapt_cov_arguments, other_arguments)

    tune_dict = tuneinput_class(input_dict).singleton_tune_dict()

    sampler1 = mcmc_sampler(tune_dict=tune_dict, mcmc_settings_dict=mcmc_meta, tune_settings_dict=tune_settings_dict)

    out = sampler1.start_sampling()
    sampler1.remove_failed_chains()

    print("num chains removed {}".format(sampler1.metadata.num_chains_removed))
    print("num restarts {}".format(sampler1.metadata.num_restarts))

    samples = sampler1.get_samples(permuted=True)

    num_mcmc_samples = samples.shape[0]
    indices = numpy.random.choice(a=num_mcmc_samples,size=num_of_pts,replace=False)

    chosen_samples = samples[indices,:]
    list_q_double = [None]*num_of_pts

    for i in range(num_of_pts):
        q_point = point(V=v_fun(precision_type="torch.DoubleTensor"))
        flattened_tensor = torch.from_numpy(chosen_samples[i,:]).type("torch.DoubleTensor")
        q_point.flattened_tensor.copy_(flattened_tensor)
        q_point.load_flatten()

        list_q_double[i] = q_point

    list_q_float = [None] * num_of_pts

    for i in range(num_of_pts):
        q_point = point(V=v_fun(precision_type="torch.FloatTensor"))
        flattened_tensor = torch.from_numpy(chosen_samples[i, :]).type("torch.FloatTensor")
        q_point.flattened_tensor.copy_(flattened_tensor)
        q_point.load_flatten()

        list_q_float[i] = q_point


    out = {"list_q_double":list_q_double,"list_q_float":list_q_float}
    return(out)





