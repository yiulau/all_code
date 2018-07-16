# functions to process output from sampler object
import math,numpy
from post_processing.ESS_nuts import ess_stan, ESJD
from post_processing.get_diagnostics import process_diagnostics
from scipy.stats import wishart as wishart
def get_ess_and_esjds(ran_sampler):
    # get max,min,median ess and normalized esjd for sampler on unconstrained space
    ran_sampler.remove_failed_chains()
    sampler_diag_check = {"num_chains_removed":ran_sampler.metadata.num_chains_removed,
                          "num_restarts":ran_sampler.metadata.num_restarts}

    if ran_sampler.metadata.num_chains_removed ==0:
        diag = ran_sampler.get_diagnostics()
        num_transitions = process_diagnostics(diag,name_list=["num_transitions"])
        total_num_transitions = sum(num_transitions)
        samples_combined = ran_sampler.get_samples(permuted=True)
        esjd  = ESJD(samples_combined)
        esjd_normalized = esjd/math.sqrt(total_num_transitions)
        ess = ess_stan(ran_sampler.get_samples(permuted=False))
        min_ess = min(ess)
        max_ess = max(ess)
        median_ess = numpy.median(ess)
        median_ess_normalized = median_ess/math.sqrt(total_num_transitions)
        min_ess_normalized = min_ess/math.sqrt(total_num_transitions)
        max_ess_normalized = max_ess/math.sqrt(total_num_transitions)

        out = {"median_ess":median_ess,"max_ess":max_ess,"min_ess":min_ess,"esjd":0,"esjd_normalized":0,"median_ess_normalized":median_ess_normalized,"min_ess_normalized":min_ess_normalized,"max_ess_normalized":max_ess_normalized}

    else:
        out = {"median_ess":0,"max_ess":0,"min_ess":0,"esjd":0,"esjd_normalized":0,"median_ess_normalized":0,"min_ess_normalized":0,"max_ess_normalized":0}

    out.update({"sampler_diag_check":sampler_diag_check})
    return(out)





def wishart_for_cov(dim):
    # returns positive definite matrix for dimension dim generated from the wishart distribution with designated
    #  degreee of freedom and identity scale matrix. recommend degree of freedom = dimension of matrix
    #
    wishart_obj = wishart()
    out = wishart.rvs(df=dim,scale=numpy.eye(dim),size=1)
    return(out)


# out = wishart_for_cov(10,10)
# #
# print(out)
# exit()


def convert_results_to_numpy_tensor():
    input_numpy_object_array = numpy.empty([2,10,3],dtype=object)

    it = numpy.nditer(input_numpy_object_array, flags=['multi_index', "refs_ok"])

    input_numpy_object_array[it.multi_index] = {"cov": numpy.random.randn(1), "sd": numpy.random.randn(1)}
    entry_names = list(input_numpy_object_array[it.multi_index].keys())
    num_items_in_entry = len(entry_names)
    new_shape = tuple(list(input_numpy_object_array.shape)+[num_items_in_entry])
    numerical_numpy_array = numpy.empty(shape=new_shape,dtype=float)

    while not it.finished:
        input_numpy_object_array[it.multi_index] = {"cov":numpy.random.randn(1),"sd":numpy.random.randn(1)}
        multi_index_in_list = list(it.multi_index)

        #print(numerical_numpy_array[tuple(multi_index_in_list+[0])])
        #exit()
        for i in range(num_items_in_entry):
            numerical_numpy_array[tuple(multi_index_in_list+[i])] = input_numpy_object_array[it.multi_index][entry_names[i]]
            numerical_numpy_array[tuple(multi_index_in_list+[i])] = input_numpy_object_array[it.multi_index][entry_names[i]]
        #print(it.multi_index)
        it.iternext()


    #save_address = "./test_array.npy"
    save_address = "./test_array.npz"
    #numpy.save(file=save_address,allow_pickle=False,arr=numerical_numpy_array)
    numpy.savez_compressed(file=save_address,numerical_array =numerical_numpy_array,names=entry_names)
    #rint(numerical_numpy_array[0,0,0,0])
    return()

#convert_results_to_numpy_tensor()

#new_array = numpy.load(file="test_array.npy")
#print(new_array[0,0,0,0])

# dict_of_arrays = numpy.load(file="test_array.npz")
#
# print(dict_of_arrays.keys())
# print(dict_of_arrays["names"])
#
# print(dict_of_arrays["numerical_array"])
