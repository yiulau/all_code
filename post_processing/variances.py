import numpy,math,time
def marginal_var(mcmc_samples_tensor):
    # warmup already discarded
    # need to split in two
    num_chains = mcmc_samples_tensor.shape[0]
    num_samples = mcmc_samples_tensor.shape[1]
    dim = mcmc_samples_tensor.shape[2]
    new_shape = (num_chains*2,num_samples//2,dim)
    num_samples = new_shape[1]
    num_chains = new_shape[0]
    split_mcmc_samples_tensor = mcmc_samples_tensor.reshape(new_shape)
    chain_means_matrix = split_mcmc_samples_tensor.sum(axis=1,keepdims=True)/num_samples
    total_means_vec = chain_means_matrix.sum(axis=0,keepdims=False)/num_chains
    chain_vars_matrix = numpy.square(split_mcmc_samples_tensor - chain_means_matrix)
    chain_vars_matrix = chain_vars_matrix.sum(axis=1,keepdims=False)/(num_samples-1)
    B_vec = num_samples*(numpy.square(chain_means_matrix - total_means_vec)).sum(axis=0,keepdims=False)/(num_chains-1)
    W_vec = chain_vars_matrix.sum(axis=0)/(num_chains)

    estimated_var_vec = (num_samples-1)*W_vec/num_samples + B_vec/num_samples
    Rhat_vec = numpy.sqrt(estimated_var_vec/W_vec)

    return(estimated_var_vec,Rhat_vec)


def single_marginal_var(sample_matrix):
    # m_ij i = sample, j = chain
    num_samples = sample_matrix.shape[0]
    num_chains = sample_matrix.shape[1]
    chain_means = sample_matrix.sum(axis=0)/num_samples

    total_mean = sum(chain_means)/num_chains

    B = num_samples*sum(numpy.square(chain_means-total_mean))/(num_chains-1)

    chain_vars = numpy.zeros(num_chains)
    for j in range(num_chains):
        temp  = 0
        for i in range(num_samples):
            temp += (sample_matrix[i,j]-chain_means[j])*(sample_matrix[i,j]-chain_means[j])
        chain_vars[j] = temp/(num_samples-1)

    W = sum(chain_vars)/num_chains

    estimated_var = (num_samples-1)*W/num_samples + B/num_samples
    rhat = math.sqrt(estimated_var/W)
    return(estimated_var,rhat)


#
# def bt_seq_var():
#     pass
#
# def in_seq_var():
#     pass
#
#
# inpu = numpy.random.randn(4,40000,200)
#
# store = [0]*2
# for i in range(len(store)):
#     inpu_m = inpu[:,:,i].reshape(8,20)
#     store[i]=single_marginal_var(inpu_m.transpose())
#
#
# print(store)

# start_time = time.time()
#
# marginal_var(inpu)
#
# print("total time {}".format(time.time()-start_time))
# exit()
#print(inpu_m)


#print(inpu)
# var = single_marginal_var(inpu_m.transpose())
#
# var2 = marginal_var(inpu)
#
# print(var)
# print(var2)
#print(marginal_var(numpy.random.randn(4,40,2)))
#inpu_m = inpu.reshape(8,200)

#var = single_marginal_var()
#print(inpu)
#original_vec = inpu[0,:,0]
#out = marginal_var(inpu)
#transformed_vec = out[0,:,0]

#print(numpy.cov(inpu.flatten()))
#var = single_marginal_var(inpu.reshape(8,200))
#print(var)
#print(out)
#print(transformed_vec)
#print(original_vec)