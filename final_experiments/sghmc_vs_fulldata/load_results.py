import numpy
results = "sghmc_results.npz"


out = numpy.load(results)


print(out.keys())

print(out["output"])