import numpy
results = "hmc_windowed_results.npz"


out = numpy.load(results)


print(out.keys())

print(out["output"].shape)


print(out["output"])