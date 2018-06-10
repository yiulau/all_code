
def block_gibbs_one_step(q,Ham):
    # q is the parameters excluding the hyperparameters
    Ham.V.load_point(q)
    Ham.V.update_hyperparam()
    out = Ham.V.get_hyperparam()
    return(out)

def update_param_and_hyperparam_one_step(init_q,init_hyperparam,Ham,epsilon,L,log_obj,param_one_step):
    Ham.V.load_hyperparam(init_hyperparam)
    out = param_one_step()
    next_q = out[0]
    next_hyperparam = block_gibbs_one_step(next_q,Ham)

    return(next_q)
