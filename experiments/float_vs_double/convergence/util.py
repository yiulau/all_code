import numpy
def convert_convergence_output_to_numpy(out):
    result = numpy.zeros(6)
    result[0] = out["diag_combined"]["min_ess"]
    result[1] = out["diag_combined"]["percent_rhat"]
    result[2] = out["diag_float"]["min_ess"]
    result[3] = out["diag_float"]["percent_rhat"]
    result[4] = out["diag_double"]["min_ess"]
    result[5] = out["diag_double"]["percent_rhat"]
    return(result)