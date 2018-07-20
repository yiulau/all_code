import numpy
results = "gibbs_v_joint_convergence_results.npz"


out = numpy.load(results)


print(out.keys())

print(out["output"])

