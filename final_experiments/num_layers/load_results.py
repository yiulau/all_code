import numpy
results = "num_layers_results.npz"


out = numpy.load(results)


print(out.keys())

print(out["output"])