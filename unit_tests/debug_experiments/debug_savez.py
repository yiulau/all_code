import numpy
results = numpy.random.randn(50,4,20)
col_names = ["num_experiments","num_chains","relevant_features"]
feature_names = ["check","two","three"]

files_dict = {"results":results,"col_names":col_names,"feature_names":feature_names}
save_address = "debug_savez"
numpy.savez(save_address,**files_dict)