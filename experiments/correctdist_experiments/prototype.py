import numpy

#from python2R.mcse_rpy2 import mcse_repy2 as mc_se
from post_processing.ESS_nuts import diagnostics_stan

# def check_mean_var(mcmc_samples,correct_mean,correct_cov,diag_only=False):
#     # mcmc_samples a numpy tensor [num_sample_per_chain,dim]
#     # expects correct_cov to be a vector when diag_only = True
#     #numpy.mean(mcmc_samples, axis=0)
#     #empCov = numpy.cov(mcmc_samples, rowvar=False)
#     #emmean = numpy.mean(mcmc_samples, axis=0)
#     #mc_se = mc_se(mcmc_samples)
#
#     if diag_only:
#         mcmc_Cov = numpy.empty(shape=[mcmc_samples.shape[1]],dtype=object)
#     else:
#         mcmc_Cov = numpy.empty(shape=[mcmc_samples.shape[1],mcmc_samples.shape[1]],dtype=object)
#     mcmc_mean = numpy.empty(shape=(mcmc_samples.shape[1]),dtype=object)
#
#     # first treat the means
#     for i in range(len(mcmc_mean)):
#         temp_vec = mcmc_samples[:,i]
#         mu = numpy.mean(temp_vec)
#         abs_diff = abs(mu - correct_mean[i])
#         MCSE = mc_se(temp_vec)
#         if abs_diff<3*MCSE:
#             reasonable = True
#         else:
#             reasonable = False
#         out = {"abs_diff":abs_diff,"MCSE":MCSE,"reasonable":reasonable}
#         mcmc_mean[i] = out
#
#
#     # treat the covariances
#     if diag_only:
#         for i in range(mcmc_Cov.shape[0]):
#             temp_vec_i = mcmc_samples[:, i]
#             var_temp_vec = numpy.square(temp_vec_i - correct_mean[i])
#             mu = numpy.mean(var_temp_vec)
#             MCSE = mc_se(var_temp_vec)
#             abs_diff = abs(mu - correct_cov[i])
#             if abs_diff < 3*MCSE:
#                 reasonable = True
#             else:
#                 reasonable = False
#             out = {"abs_diff": abs_diff, "MCSE": MCSE, "reasonable": reasonable}
#             mcmc_Cov[i] = out
#     else:
#         for i in range(mcmc_Cov.shape[0]):
#             for j in range(mcmc_Cov.shape[1]):
#                 if not i==j:
#                     temp_vec_i = mcmc_samples[:,i]
#                     temp_vec_j = mcmc_samples[:,j]
#                     #covar_temp_vec = (temp_vec_i - correct_mean[i])*(temp_vec_j-correct_mean[j])/\
#                     #                (numpy.sqrt(correct_cov[i,i]*correct_cov[j,j]))
#                     covar_temp_vec = (temp_vec_i - correct_mean[i])*(temp_vec_j-correct_mean[j])
#                     mu = numpy.mean(covar_temp_vec)
#                     MCSE = mc_se(covar_temp_vec)
#                     abs_diff = abs(mu-correct_cov[i,j])
#                 else:
#                     temp_vec_i = mcmc_samples[:, i]
#                     var_temp_vec = numpy.square(temp_vec_i - correct_mean[i])
#                     mu = numpy.mean(var_temp_vec)
#                     MCSE = mc_se(var_temp_vec)
#                     abs_diff = abs(mu-correct_cov[i,i])
#                 if abs_diff < 3*MCSE:
#                     reasonable = True
#                 else:
#                     reasonable = False
#                 out = {"abs_diff":abs_diff,"MCSE":MCSE,"reasonable":reasonable}
#                 mcmc_Cov[i,j]=out
#
#     denom = 0.
#     num = 0.
#     for i in range(len(mcmc_mean)):
#         num +=float(mcmc_mean[i]["reasonable"])
#         denom +=1
#     pc_of_mean = num/denom
#     num = 0.
#     denom = 0.
#     if diag_only:
#         for i in range(mcmc_Cov.shape[0]):
#             num += float(mcmc_Cov[i]["reasonable"])
#             denom +=1
#     else:
#         for i in range(mcmc_Cov.shape[0]):
#             for j in range(mcmc_Cov.shape[1]):
#                 num += float(mcmc_Cov[i,j]["reasonable"])
#                 denom +=1
#     pc_of_cov = num/denom
#
#     out = {"mcmc_mean":mcmc_mean,"mcmc_Cov":mcmc_Cov,"pc_of_mean":pc_of_mean,"pc_of_cov":pc_of_cov}
#     return(out)



def check_mean_var_stan(mcmc_samples, correct_mean, correct_cov, diag_only=False):
    # mcmc_samples a numpy tensor [num_chains,num_sample_per_chain,dim]
    # expects correct_cov to be a vector when diag_only = True
    # numpy.mean(mcmc_samples, axis=0)
    # empCov = numpy.cov(mcmc_samples, rowvar=False)
    # emmean = numpy.mean(mcmc_samples, axis=0)
    # mc_se = mc_se(mcmc_samples)

    num_samples_per_chain =  mcmc_samples.shape[1]
    chain_combined_mcmc_samples = mcmc_samples.reshape(-1,mcmc_samples.shape[2])
    if diag_only:
        mcmc_Cov = numpy.empty(shape=[mcmc_samples.shape[2]], dtype=object)
    else:
        mcmc_Cov = numpy.empty(shape=[mcmc_samples.shape[2], mcmc_samples.shape[2]], dtype=object)
    mcmc_mean = numpy.empty(shape=(mcmc_samples.shape[2]), dtype=object)

    mcmc_mean = numpy.mean(chain_combined_mcmc_samples,axis=0)
    mean_diagnostics = diagnostics_stan(mcmc_samples)
    mcmc_mean_mcse = mean_diagnostics["mcse"]
    mcmc_mean_rhat = mean_diagnostics["rhat"]
    mean_abs_diff =  abs(mcmc_mean-correct_mean)

    # first treat the means
    mean_reasonable = mean_abs_diff < 3 * mcmc_mean_mcse
    mean_results = numpy.empty(len(mcmc_mean),dtype=object)
    #print(mean_reasonable)
    #print(mcmc_mean_mcse)
    #print(mcmc_mean_rhat)
    for i in range(len(mean_results)):


        out = {"abs_diff": mean_abs_diff[i], "MCSE": mcmc_mean_mcse[i],"rhat":mcmc_mean_rhat[i], "reasonable": mean_reasonable[i]}
        mean_results[i] = out

    # treat the covariances
    if diag_only:
        var_temp  = numpy.square(mcmc_samples - correct_mean)
        diag_var_diagnostics = diagnostics_stan(var_temp)
        mcmc_diag_var_mcse = diag_var_diagnostics["mcse"]
        mcmc_diag_var_rhat = diag_var_diagnostics["rhat"]
        var_temp_flattened = numpy.square(chain_combined_mcmc_samples-correct_mean)
        mu = numpy.mean(var_temp_flattened,axis=0)
        diag_var_abs_diff = abs(mu-correct_cov)
        var_reasonable = diag_var_abs_diff < 3 * mcmc_diag_var_mcse
        for i in range(mcmc_Cov.shape[0]):

            out = {"abs_diff": diag_var_abs_diff[i], "MCSE": mcmc_diag_var_mcse[i],"rhat":mcmc_diag_var_rhat[i],
                   "reasonable": var_reasonable[i]}
            mcmc_Cov[i] = out
    else:
        for i in range(mcmc_Cov.shape[0]):
            for j in range(mcmc_Cov.shape[1]):
                if not i == j:
                    temp_vec_i = mcmc_samples[:,:, i:i+1]
                    temp_vec_j = mcmc_samples[:,:, j:j+1]
                    # covar_temp_vec = (temp_vec_i - correct_mean[i])*(temp_vec_j-correct_mean[j])/\
                    #                (numpy.sqrt(correct_cov[i,i]*correct_cov[j,j]))
                    #print(temp_vec_i.shape)
                    #print(temp_vec_j.shape)
                    covar_temp_vec = (temp_vec_i - correct_mean[i]) * (temp_vec_j - correct_mean[j])
                    cov_ij_diagnostics = diagnostics_stan(covar_temp_vec)
                    mu = numpy.mean(covar_temp_vec.flatten())
                    covar_mcse = cov_ij_diagnostics["mcse"]
                    covar_rhat = cov_ij_diagnostics["rhat"]
                    abs_diff = abs(mu - correct_cov[i, j])

                else:
                    temp_vec_i = mcmc_samples[:,:,i:i+1]
                    var_temp_vec = numpy.square(temp_vec_i - correct_mean[i])
                    mu = numpy.mean(var_temp_vec.flatten())
                    var_temp_diagnostics = diagnostics_stan(var_temp_vec)
                    covar_mcse = var_temp_diagnostics["mcse"]
                    covar_rhat = var_temp_diagnostics["rhat"]
                    abs_diff = abs(mu - correct_cov[i, i])
                reasonable = abs_diff < 3 * covar_mcse

                out = {"abs_diff": abs_diff, "MCSE": covar_mcse,"rhat":covar_rhat, "reasonable": reasonable}
                mcmc_Cov[i, j] = out

    denom = 0.
    num = 0.
    for i in range(len(mean_results)):
        num += float(mean_results[i]["reasonable"])
        denom += 1
    pc_of_mean = num / denom
    num = 0.
    denom = 0.
    if diag_only:
        for i in range(mcmc_Cov.shape[0]):
            num += float(mcmc_Cov[i]["reasonable"])
            denom += 1
    else:
        for i in range(mcmc_Cov.shape[0]):
            for j in range(mcmc_Cov.shape[1]):
                num += float(mcmc_Cov[i, j]["reasonable"])
                denom += 1
    pc_of_cov = num / denom

    out = {"mcmc_mean": mean_results, "mcmc_Cov": mcmc_Cov, "pc_of_mean": pc_of_mean, "pc_of_cov": pc_of_cov}
    return (out)


#
# correct_mean = numpy.zeros(2)
# correct_cov = numpy.eye(2)
# input_tensor = numpy.random.randn(1,1000,2)
# out = check_mean_var_stan(input_tensor,correct_mean,correct_cov)
#
# print(out["mcmc_mean"])
#
# print(out["mcmc_Cov"])
#
# print(out)
#
#
