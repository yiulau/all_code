import numpy
unscaled_name = "ensemble_results.npz"

out = numpy.load(unscaled_name)

print(out.keys())

print(out["output"])