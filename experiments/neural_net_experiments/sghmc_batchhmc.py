#
import torch
def sghmc_one_step(init_q_point,epsilon,L,Ham,alpha,eta,betahat,input_data):
    # eta is the learning rate
    init_v_point = init_q_point.point_clone()

    v = init_v_point.point_clone()
    q = init_q_point.point_clone()
    dim = len(init_q_point.flattened_tensor)
    v.flattened_tensor.copy_(torch.randn(dim)*epsilon)

    for i in range(L):


        q.flattened_tensor += v.flattened_tensor
        q.load_flatten()
        noise = torch.randn(dim)

        delta_v = -eta*Ham.V.dq(input_data,q) - alpha * v.flattened_tensor + torch.sqrt(2*(alpha-betahat)*noise)
        v.flattened_tensor.copy_(delta_v)
        v.load_point()
    return(q)


def sghmc_sampler(init_q_point,epsilon,L,Ham,alpha,eta,betahat,full_data,num_samples,thin,burn_in):
    dim = len(init_q_point.flattened_tensor)
    store= torch.zeros(round(num_samples/thin),dim)
    cur = thin
    store_i = 0
    q = init_q_point.point_clone()
    for i in range(num_samples):
        input_data = full_data.randombatch()
        q = sghmc_one_step(q,epsilon,L,Ham,alpha,eta,betahat,input_data)
        if i >= burn_in:
            cur -= 1
            if not cur > 0.1:
                keep = True
                store_i += 1
                cur = thin
            else:
                keep = False
            store[store_i-1,:].copy_(q.flattened_tensor.copy())

    return(store)







