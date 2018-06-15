# functions to process output from sampler object
import math,numpy
from adapt_util.objective_funs_util import ESJD
from post_processing.ESS_nuts import ess_stan
from scipy.stats import wishart as wishart
def get_min_ess_and_esjds(ran_sampler):

    ran_sampler.remove_failed_chains()
    sampler_diag_check = {"num_chains_removed":ran_sampler.metadata.num_chains_removed,
                          "num_restarts":ran_sampler.metadata.num_restarts}

    if ran_sampler.metadata.num_chains_removed ==0:
        samples_combined = ran_sampler.get_samples(permuted=True)
        esjd  = ESJD(samples_combined)
        esjd_normalized = esjd/math.sqrt(ran_sampler.metadata.average_num_transitons)
        ess = ess_stan(ran_sampler.get_samples(permuted=False))
        min_ess = min(ess)

        out = {"min_ess":min_ess,"esjd":esjd,"esjd_normalized":esjd_normalized}

    else:
        out = {"min_ess":0,"esjd":0,"esjd_normalized":0}

    out.update({"sampler_diag_check":sampler_diag_check})
    return(out)





def wishart_for_cov(dim,degree_freedom):
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

dict_of_arrays = numpy.load(file="test_array.npz")

print(dict_of_arrays.keys())
print(dict_of_arrays["names"])

print(dict_of_arrays["numerical_array"])
