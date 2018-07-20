import numpy
stability_results = "float_vs_double_stability_results.npz"

test_error_results = "float_vs_double_convergence_results.npz"

out = numpy.load(stability_results)

out_te = numpy.load(test_error_results)

print(out_te.keys())
print(out.keys())

print(out["output"])

print(out_te["output"])
