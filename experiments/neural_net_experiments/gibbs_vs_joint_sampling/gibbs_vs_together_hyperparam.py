from abstract.abstract_static_sampler import abstract_static_one_step
from abstract.abstract_nuts_util import abstract_GNUTS

def block_gibbs_one_step(q,Ham):
    # q is the parameters excluding the hyperparameters
    Ham.V.load_point(q)
    Ham.V.update_hyperparam()
    out = Ham.V.get_hyperparam()
    return(out)


def update_param_and_hyperparam_one_step(init_q,init_hyperparam,Ham,epsilon,L,log_obj):
    Ham.V.load_hyperparam(init_hyperparam)
    out = abstract_static_one_step(epsilon=epsilon,init_q=init_q,Ham=Ham,evolve_L=L,log_obj=log_obj)
    next_q = out[0]
    next_hyperparam = block_gibbs_one_step(next_q,Ham)
    return(next_q,next_hyperparam)


def update_param_and_hyperparam_dynamic_one_step(init_q,init_hyperparam,Ham,epsilon,log_obj):
    Ham.V.load_hyperparam(init_hyperparam)
    out = abstract_GNUTS(epsilon=epsilon,init_q=init_q,Ham=Ham,log_obj=log_obj)
    next_q = out[0]
    next_hyperparam = block_gibbs_one_step(next_q,Ham)
    return(next_q,next_hyperparam)



