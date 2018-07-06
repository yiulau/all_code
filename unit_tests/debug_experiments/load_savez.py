import os.path
import numpy
save_address = "debug_savez.npz"
if os.path.isfile(save_address) :
    out = numpy.load(save_address)

print(out.files)
print(out["results"])
print(out["col_names"])
print(out["feature_names"])
