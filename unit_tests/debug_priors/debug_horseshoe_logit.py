import numpy

# perfect separation logit data

n =30
dim= 100
X = numpy.random.randn(n,dim)
y = [None]*n
for i in range(n):
    y[i] = numpy.asscalar(numpy.random.choice(2,1))
    if y[i] > 0:
        X[i,0:2] = numpy.random.randn(2)*0.5 + 1
    else:
        X[i,0:2] = numpy.random.randn(2)*0.5 -1


print(type(y[0]))