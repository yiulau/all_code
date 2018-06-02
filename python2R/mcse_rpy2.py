import readline
import rpy2
#import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
import numpy
from rpy2 import robjects
#pi = robjects.r['pi']

mcmcse = importr("mcmcse")
#ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
#out = mcmcse.mcse(ctl)
#print(out.names)
#print(type(out[1]))
#out = (numpy.asarray(out[1]))
#print(pi[0])

def mcse_repy2(numpy_vec):
    ctl = FloatVector(numpy_vec)
    out = mcmcse.mcse(ctl)
    out = numpy.asarray(out[1])
    return(out)

def ess_mcse_repy2(numpy_matrix):
    nrow = numpy_matrix.shape[0]
    ctl = robjects.r.matrix(FloatVector(numpy_matrix.flatten()), nrow=nrow)
    out = mcmcse.ess(ctl)
    print(out)
    out = numpy.asarray(out)
    return(out)
x=numpy.random.randn(10000,10)

#print(ess_mcse_repy2(x))
#exit()
#mcse_repy2(x[:,0])
#for i in range(1000):
#    print("counter {}".format(i))
#    print(mcse_repy2(x[:, i]))
#print(x)

# 4250.60717606  302.85180722 1873.47052937 3532.24607792  583.61198642
#  1261.66650871  298.95940212

# 934.62678222  823.58491743 1727.90051369 2043.53974196  358.26480736
#   445.48189914 2806.03486298