library(reticulate)

# load numpy matrix
numpy = import("numpy")
X = numpy$load("input_matrix.npy")

print(X[2,5,4])

# save numpy matrix

outX = array(rnorm(10000),dim=c(2,5,1000))

numpy$save("matrix_from_R.npy",arr=outX)
