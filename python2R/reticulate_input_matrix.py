import numpy

numpy.random.seed(1)

X = numpy.random.randn(100,10,5)

print(X[1,4,3])
numpy.save("input_matrix.npy",arr=X,allow_pickle=False)

#from_R = numpy.load("matrix_from_R.npy")

#print(from_R.shape)