
def block_gibbs_one_step(q,Ham):
    # q is the parameters excluding the hyperparameters
    Ham.V.load_point(q)
    Ham.V.update_hyperparam()

    return()


