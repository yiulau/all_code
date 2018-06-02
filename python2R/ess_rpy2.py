import readline
import rpy2
#import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2 import robjects
import numpy
coda = importr("coda")

def ess_repy2(numpy_matrix):
    nrow = numpy_matrix.shape[0]
    ctl = robjects.r.matrix(FloatVector(numpy_matrix.flatten()), nrow=nrow)
    out = coda.mcmc(ctl)
    ess = coda.effectiveSize(out)

    ess = numpy.asarray(ess)
    return(ess)


# tryinput = numpy.random.randn(120,3)
#
# out = ess_repy2(tryinput)
#
# print(out)
