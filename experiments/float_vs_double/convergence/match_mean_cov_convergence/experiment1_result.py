import numpy

names_list = ["mvn","pima_indian"]

for i in range(len(names_list)):
    save_name = "match_convergence_result_{}.npz".format(names_list[i])
    out = numpy.load(save_name)
    print(names_list[i])
    #print(out.keys())
    print(out.items())