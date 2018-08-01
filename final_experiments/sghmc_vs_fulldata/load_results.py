import numpy
results = "sghmc_results.npz"


out = numpy.load(results)


print(out.keys())

print(out["output"].shape)

#print(out["output"])
print(out["output"][3,3,:,:])