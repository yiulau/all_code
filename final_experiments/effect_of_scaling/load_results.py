import numpy
scaled_name = "scaled_results.npz"
unscaled_name = "unscaled_results.npz"
out_scaled = numpy.load(scaled_name)
out_unscaled = numpy.load(unscaled_name)

print(out_scaled.keys())

print(out_scaled["output"])

print(out_unscaled["output"])

print(out_scaled["diagnostics"].shape)

print(out_scaled["diagnostics_names"])

print(out_scaled["diagnostics"][0,0,10:])

print(out_unscaled["diagnostics"][0,0,10:])
