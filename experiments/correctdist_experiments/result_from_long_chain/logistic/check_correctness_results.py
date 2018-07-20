import numpy

names_list = ["pima_indian","australian","german","heart","breast"]
for i in range(len(names_list)):
    save_name = "check_result_{}.npz".format(names_list[i])

    results=  numpy.load(save_name)

    print(results["pc_of_mean"])
    print(results["pc_of_cov"])