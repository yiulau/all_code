import numpy
unscaled_name = "scaled_results.npz"

out = numpy.load(unscaled_name)

print(out.keys())

print(out["output"])
