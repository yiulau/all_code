import numpy,statsmodels
from post_processing.variances import marginal_var
from scipy.signal import fftconvolve

def ess_nuts(mcmc_samples,correct_samples):
    correct_mean = numpy.mean(correct_samples,axis=0)
    correct_var = numpy.cov(correct_samples,rowvar=False).diagonal()
    M = mcmc_samples.shape[0]
    M_cutoff_vec = numpy.zeros(mcmc_samples.shape[1])
    store_lags = numpy.zeros(mcmc_samples.shape)
    ess_out = numpy.zeros(mcmc_samples.shape[1])
    s = 0
    keep_going = True
    while keep_going:
        rho_s = lag_nuts(mcmc_samples,correct_mean,correct_var)
        store_lags[s,:] = rho_s
        keep_going_vec = rho_s < 0.05
        M_cutoff_vec += keep_going_vec
        keep_going = (sum(keep_going_vec) > 0)
        if keep_going:
            s += 1
    for j in range(len(ess_out)):
        M_cutoff = M_cutoff_vec
        temp = numpy.dot((1 - numpy.array(range(M_cutoff))/M),store_lags[:M_cutoff,j])
        ess_out[j] = M/(1+2*temp)
    return(ess_out)

def lag_nuts(target_matrix,mean_vec,var_vec,s):
    # each column a parameter
    # mean_vec mean vector from the true distribution / a long converged chain
    # var_vec variance vector from the true dist/ a long chain
    # lag s
    M = target_matrix.shape[0]
    out = numpy.zeros(target_matrix.shape[1])
    for i in range(s,M):
        out += (target_matrix[i,:] - mean_vec)*(target_matrix[i-s,:]-mean_vec)

    out *= 1/(var_vec*(M-s))
    return(out)


def lag_stan(mcmc_samples_tensor):
    M = mcmc_samples_tensor.shape[0]
    acorr = 0
    return(acorr)

import time

def variogram(mcmc_samples_tensor):
    M = mcmc_samples_tensor.shape[0]
    dim = mcmc_samples_tensor.shape[2]
    num_samples_chain = mcmc_samples_tensor.shape[1]
    store_variogram = numpy.zeros((dim,num_samples_chain))
    for k in range(dim):
        temp = numpy.zeros(num_samples_chain)
        for i in range(M):
            vec = mcmc_samples_tensor[i,:,k]
            this_sum_vec = core_sum(vec)
            temp += this_sum_vec
        store_variogram[k,:]= temp
    out = store_variogram/(M*(num_samples_chain-numpy.array(range(num_samples_chain))))
    return(out)



def core_sum(vec):
    # sum_{n=t+1}^N (theta_n - theta_(n-t))^2
    # output vector of length len(vec)
    vec_squared = vec * vec
    out_alt = fftconvolve(vec_squared + vec_squared[::-1], numpy.ones(len(vec_squared)))
    out_alt = out_alt[len(out_alt) // 2:]
    out3 = fftconvolve(vec, vec[::-1])
    out3 = out3[len(out3) // 2:]
    out = out_alt -2 *out3
    return(out)


def core_sum_brute_force(inp):
    final_out_brute_force = [0] * len(inp)
    for j in range(len(inp)):
        temp = 0
        for i in range(j, len(inp)):
            temp += (inp[i] - inp[i - j]) * (inp[i] - inp[i - j])
        final_out_brute_force[j] = temp
    return(final_out_brute_force)

def variogram_brute_force(inp):
    out_bf = numpy.zeros((inp.shape[2], inp.shape[1]))
    M = inp.shape[0]
    num_samples = inp.shape[1]
    dim = inp.shape[2]
    for i in range(M):
        for j in range(dim):
            vec = inp[i, :, j]
            for t in range(num_samples):
                temp = 0
                for cur in range(t, num_samples):
                    temp += (vec[cur] - vec[cur - t]) * (vec[cur] - vec[cur - t])
                out_bf[j, t] += temp / (num_samples - t)

    out_bf = out_bf / M
    return(out_bf)


def ess_stan(mcmc_samples_tensor):
    #mcmc_samples_tensor = split_chains(mcmc_samples_tensor)
    vario_g = variogram(mcmc_samples_tensor)
    vars = marginal_var(mcmc_samples_tensor).transpose()
    store_lags = 1 - vario_g/(2*vars)
    sum_lags = numpy.zeros(store_lags.shape[0])
    for i in range(len(sum_lags)):
        temp = 0
        for cur in range(store_lags.shape[1]//2-1):
            sum_lag = store_lags[i,2*cur]+store_lags[i,2*cur+1]
            if sum_lag< 0 :
                print(cur)
                break
            else:
                temp += sum_lag
        sum_lags[i] = temp

    M = mcmc_samples_tensor.shape[0]
    num_samples = mcmc_samples_tensor.shape[1]
    ess_vec = M*num_samples/(1+2*sum_lags)
    return(ess_vec)

def split_chains(mcmc_samples_tensor):
    # assume input dimension is [num_chains,num_samples_per_chain,model_dim]
    # output dimension is [2*num_chains,num_samples_per_chain/2,model_dim]
    # cut each chain in half to double the number of chains and decrease the number of samples per chain by half
    num_chains = mcmc_samples_tensor.shape[0]
    num_samples_per_chain = mcmc_samples_tensor.shape[1]
    new_num_samples_per_chain = round(num_samples_per_chain/2)
    dim = mcmc_samples_tensor.shape[2]
    new_mcmc_samples_tensor = numpy.zeros((2*num_chains,new_num_samples_per_chain,dim))
    for i in range(num_chains):
        original_chain = mcmc_samples_tensor[i,:,:]
        new_mcmc_samples_tensor[2*i,:new_num_samples_per_chain,:] = mcmc_samples_tensor[i,:new_num_samples_per_chain,:]
        new_mcmc_samples_tensor[2*i+1,:new_num_samples_per_chain,:]=\
            mcmc_samples_tensor[i,new_num_samples_per_chain:2*new_num_samples_per_chain,:]
#
    return(new_mcmc_samples_tensor)



def diagnostics_stan(mcmc_samples_tensor):
    mcmc_samples_tensor = split_chains(mcmc_samples_tensor)
    vario_g = variogram(mcmc_samples_tensor)
    var,Rhat = marginal_var(mcmc_samples_tensor)
    vars = numpy.expand_dims(var,axis=1)
    store_lags = 1 - vario_g/(2*vars)
    sum_lags = numpy.zeros(store_lags.shape[0])
    for i in range(len(sum_lags)):
        temp = 0
        for cur in range(store_lags.shape[1]//2-1):
            sum_lag = store_lags[i,2*cur]+store_lags[i,2*cur+1]
            if sum_lag< 0 :
                #print(cur)
                break
            else:
                temp += sum_lag
        sum_lags[i] = temp

    M = mcmc_samples_tensor.shape[0]
    num_samples = mcmc_samples_tensor.shape[1]
    ess_vec = M*num_samples/(1+2*sum_lags)
    sd = numpy.sqrt(var.squeeze())
    mcse = sd/numpy.sqrt(ess_vec)
    out = {"ess":ess_vec,"sd":sd,"mcse":mcse,"rhat":Rhat}
    return(out)



# def diagnostics_stan(mcmc_samples_tensor):
#     # compute ess, mcse and Rhat

# inpu_tensor = numpy.random.randn(4,10000,5)
#
# out = diagnostics_stan(inpu_tensor)
# #
# print(out)
# exit()
#new_input_tensor = split_chains(inpu_tensor)
#print(new_input_tensor.shape)
#exit()
# inpu_tensor = numpy.random.randn(4,1000,1)
#
# out = ess_stan(inpu_tensor)
# print(out)
# exit()
# #inpu_tensor = inpu.reshape((2,1000,3))
# #o1 = core_sum(inpu)
# o2 = variogram(inpu_tensor)
# o3 = variogram_brute_force(inpu_tensor)
# #o2 = o2[0,:]
# #o3 = o3[0,:]
# #diff = sum(o1-o2)
# diff = (o2-o3).sum()
# print("diff {}".format(diff))
# exit()
#
#
# o1 = core_sum(inpu)
# o2 = core_sum_brute_force(inpu)
# print("diff core sum {}".format(sum(o1-o2)))
# inpu = numpy.random.randn(2,1000,3)
# o1 = variogram()
#
#
#
#
#
#
#
# inp = numpy.random.randn(1,500,1)
# total_time = time.time()
# outfft = variogram(inp)
# print("time {}".format(time.time()-total_time))
# #exit()
# #print(outfft)
# #exit()
#
#
#
#
# #diff = (out_bf - outfft).sum()
#
# print("diff bf fft {}".format(diff))
# #exit()
#
#
# # for k in range(inp.shape[2]):
# #     temp_here = numpy.zeros(inp.shape[1])
# #     for i in range(inp.shape[0]):
# #         temp_inp = inp[i,:,k]
# #         temp_out = numpy.zeros(len(temp_inp))
# #         for j in range(len(temp_inp)):
# #             temp = 0
# #             for i in range(j, len(temp_inp)):
# #                 temp += (temp_inp[i] - temp_inp[i - j]) * (temp_inp[i] - temp_inp[i - j])
# #             temp_out[j] = temp
# #         temp_here += temp_out/(len(temp_inp)-numpy.array(range(len(temp_inp))))
# #     out_bf[k,:] = temp_here
#
# # out_bf = out_bf * (1/inp.shape[1])
# #
# # diff_variogram = (out_bf-outfft).sum()
# # print("diff between bf and fft {}".format(diff_variogram))
# exit()
# inp = inp[-1,:,-1]
#
#
# inp_squared = numpy.square(inp)
# # this calculates sum_t+1^ N  squared(theta_n)
# ffttime = time.time()
# out = fftconvolve(inp_squared,numpy.ones(len(inp_squared)))
# out = out[len(out)//2:]
#
# out2 = fftconvolve(inp_squared[::-1],numpy.ones(len(inp_squared)))
# out2 = out2[len(out2)//2:]
#
# out3 = fftconvolve(inp,inp[::-1])
# out3 = out3[len(out3)//2:]
#
# final_out = out + out2 - 2*out3
#
# ffttime = time.time() - ffttime
#
# print("fftime {}".format(ffttime))
#
# fftime_alt = time.time()
# out_alt = fftconvolve(inp_squared+inp_squared[::-1],numpy.ones(len(inp_squared)))
# out_alt = out_alt[len(out_alt)//2:]
# out3 = fftconvolve(inp,inp[::-1])
# out3 = out3[len(out3)//2:]
#
# fftime_alt = time.time() - fftime_alt
# final_out_alt = out_alt -2 *out3
#
# print("final time alt {}".format(fftime_alt))
# #print(sum(out_alt-out-out2))
# #print(sum(final_out_alt-final_out))
#
# bf_time = time.time()
#
# out_brute_force = [0]*len(inp)
# for j in range(len(inp)):
#     temp = 0
#     for i in range(j,len(inp_squared)):
#         temp += inp_squared[i]
#     out_brute_force[j]=temp
#
#
#
# out_brute_force2 = [0]*len(inp)
# for j in range(len(inp)):
#     temp = 0
#     for i in range(j,len(inp_squared)):
#         temp += inp_squared[i-j]
#     out_brute_force2[j]=temp
#
# out_brute_force3 = [0]*len(inp)
# for j in range(len(inp)):
#     temp = 0
#     for i in range(j,len(inp_squared)):
#         temp += inp[i]*inp[i-j]
#     out_brute_force3[j] = temp
#
# final_out_brute_force = [0]*len(inp)
# for j in range(len(inp)):
#     temp = 0
#     for i in range(j,len(inp)):
#         temp += (inp[i] - inp[i-j])*(inp[i]-inp[i-j])
#     final_out_brute_force[j] = temp
#
#
# print(sum(final_out_brute_force-final_out))
#
# print(sum(out3-out_brute_force3))
#
# print(sum(out_brute_force2-out2))
#
# print(sum(out_brute_force - out))
#
#
# print("fftime {}".format(ffttime))
# print("bf time {}".format(bf_time))
# exit()
# print(len(out))
# exit()
# lag_stan(inp)
# exit()
# print(lag_stan(inp))

