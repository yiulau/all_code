import numpy
unscaled_name = "effects_prior_results.npz"

out = numpy.load(unscaled_name)

print(out.keys())

print(out["output"])
