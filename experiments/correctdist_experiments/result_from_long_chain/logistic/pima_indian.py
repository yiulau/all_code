from experiments.correctdist_experiments.result_from_long_chain.logistic.util import result_from_long_chain
from input_data.convert_data_to_dict import get_data_dict
import numpy

names_list = ["pima_indian","australian","german","heart","breast"]

for i in range(len(names_list)):
    input_data = get_data_dict(names_list[i])

    out = result_from_long_chain(input_data=input_data,data_name=names_list[i],recompile=False)

    file = numpy.load(out)

#print(file.keys())

    print(file["correct_mean"])
    print(file["correct_cov"])


