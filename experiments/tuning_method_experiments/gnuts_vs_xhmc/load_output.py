import numpy
save_name = "weak_correlation.npz"

out_weak = numpy.load(save_name)
print(out_weak.files)

print(out_weak["gnuts_diagnostics"].shape)
print(out_weak["xhmc_diagnostics"].shape)
print(out_weak["diagnostics_names"])
print(out_weak["gnuts_result"].shape)
print(out_weak["xhmc_result"].shape)

print(out_weak["gnuts_col_names"])

# print(out_weak["xhmc_col_names"])
#
# print(out_weak["gnuts_output_names"])
# print(out_weak["xhmc_output_names"])

print(out_weak["gnuts_diagnostics"][...,7])
print(out_weak["gnuts_diagnostics"][...,4])