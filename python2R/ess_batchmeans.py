import readline
import rpy2
#import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2 import robjects
import numpy
batchmeans = importr("batchmeans")

def ess_batchmeans_repy2(numpy_matrix):
    #nrow = numpy_matrix.shape[0]
    #FloatVector(numpy_vec)
    #ctl = robjects.r.matrix(FloatVector(numpy_matrix.flatten()), nrow=nrow)
    output = []
    for i in range(numpy_matrix.shape[1]):
        ctl = FloatVector(numpy_matrix[:,i])
        out = batchmeans.ess(ctl)
        output.append(numpy.asarray(out))


    #ess = numpy.asarray(ess)
    return(output)

#tryinput = numpy.random.randn(1200,3)
#out = ess_batchmeans_repy2(tryinput)
#print(out)

# [array([157.59206583]), array([248.05834637]), array([468.17510292]), array([141.36686068]), array([109.04960112]), array([490.]), array([76.78016491])]
