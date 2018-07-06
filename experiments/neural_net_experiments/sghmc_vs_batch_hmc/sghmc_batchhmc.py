#
import torch, numpy,math
def sghmc_one_step(init_q_point,epsilon,L,Ham,alpha,eta,betahat,input_data,adjust_factor):
    # eta is the learning rate
    # adjust_factor should be full_data_size/batch_size
    #print(adjust_factor)
    #exit()
    init_v_point = init_q_point.point_clone()

    v = init_v_point.point_clone()
    q = init_q_point.point_clone()
    dim = len(init_q_point.flattened_tensor)
    v.flattened_tensor.copy_(torch.randn(dim)*epsilon)
    explode_grad = False

    for i in range(L):
        q.flattened_tensor += v.flattened_tensor
        q.load_flatten()
        noise = torch.randn(dim)
        grad,explode_grad = Ham.V.dq(q_flattened_tensor=q.flattened_tensor,input_data=input_data)
        v_val = Ham.V.forward()
        #print("v {}".format(v_val))
        if not explode_grad:
            grad = grad*adjust_factor
            delta_v = -eta*grad - alpha * v.flattened_tensor + math.sqrt(2*(alpha-betahat))*noise
            v.flattened_tensor += delta_v
            v.load_flatten()
        else:
            break

    return(q,explode_grad)


def sghmc_sampler(init_q_point,epsilon,L,Ham,alpha,eta,betahat,full_data,num_samples,thin,burn_in,batch_size):
    dim = len(init_q_point.flattened_tensor)
    full_data_size = len(full_data["target"])
    full_data = {"input":torch.from_numpy(full_data["input"]),"target":torch.from_numpy(full_data["target"])}
    if thin>0:
        store= torch.zeros(round(num_samples/thin),dim)
    else:
        assert thin==0
        store = torch.zeros(num_samples,dim)
    cur = thin
    store_i = 0
    explode_grad = False
    q = init_q_point.point_clone()
    for i in range(num_samples):
        input_data = subset_data(full_data=full_data,batch_size=batch_size)
        q,explode_grad = sghmc_one_step(q,epsilon,L,Ham,alpha,eta,betahat,input_data,full_data_size/batch_size)
        print(q.flattened_tensor)
        if not explode_grad:
            if i >= burn_in:
                cur -= 1
                if not cur > 0.1:
                    keep = True
                    store_i += 1
                    cur = thin
                else:
                    keep = False
                store[store_i-1,:].copy_(q.flattened_tensor.clone())
        else:
            break

    return(store,explode_grad)




def subset_data(full_data,batch_size):
    full_data_size = len(full_data["target"])
    chosen_indices = list(numpy.random.choice(a=full_data_size, size=batch_size, replace=False))
    chosen_indices = [numpy.asscalar(v) for v in chosen_indices]
    subset_input = full_data["input"][chosen_indices,:]
    subset_target = full_data["target"][chosen_indices]
    out = {"input":subset_input,"target":subset_target}
    return(out)


