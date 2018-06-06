
def block_gibbs_one_step(q,Ham):

    Ham.V.load_point(q)
    update_indices = Ham.V.hyper_param_indices
    for i in range(update_indices):
        new_val = Ham.V.get_update_hyperparam_val(i)
        q.list_tensor[i].copy_(new_val)

    q.load_param_to_flattened()
    return(q)


