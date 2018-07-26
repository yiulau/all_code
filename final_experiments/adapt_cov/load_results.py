import numpy
unscaled_name = "adapt_cov_results.npz"

out = numpy.load(unscaled_name)

print(out.keys())

print(out["output"])
