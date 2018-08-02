import numpy
stability_results = "float_vs_double_stability_results.npz"

test_error_results = "float_vs_double_convergence_results.npz"

out = numpy.load(stability_results)

out_te = numpy.load(test_error_results)

#print(out_te.keys())
# print(out.keys())
#
# print(out["output"].shape)
#
# float1 = out["output"][0,0,:]
# double1 = out["output"][0,1,:]
# float2 = out["output"][1,0,:]
# double2 = out["output"][1,1,:]
# print(numpy.mean(float1-double1))
# print(numpy.mean(float2-double2))
print(out_te["output"])
