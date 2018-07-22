import numpy
results = "xhmc_gnuts_results.npz"


out = numpy.load(results)


print(out.keys())

print(out["output"])

print(out["diagnostics"].shape)

print(out["diagnostics"][:,:,:4])
