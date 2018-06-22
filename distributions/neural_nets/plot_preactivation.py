import numpy
# plot preactivation distributions for a single unit for different priors

num_samples = 5000

input_dim = 100
# normal prior
store_pre_activation = numpy.zeros(num_samples)

for i in range(num_samples):
    X_input = numpy.random.randn(input_dim)
    beta = numpy.random.randn(input_dim) * numpy.sqrt(1/input_dim)
    pre_activation = numpy.dot(X_input,beta)
    store_pre_activation[i] = pre_activation

mean = numpy.mean(store_pre_activation)
var = numpy.var(store_pre_activation)

print("normal prior induced mean {}".format(mean))
print("normal prior induced var {}".format(var))

# student t prior - df = 1
store_pre_activation = numpy.zeros(num_samples)

for i in range(num_samples):
    X_input = numpy.random.randn(input_dim)
    df = 1.5
    precision = numpy.random.gamma(shape=0.5*df,scale=1/(df*0.5))
    beta = numpy.random.randn(input_dim) * numpy.sqrt(2/input_dim) * numpy.sqrt(1/precision)
    pre_activation = numpy.dot(X_input,beta)
    store_pre_activation[i] = pre_activation

mean = numpy.mean(store_pre_activation)
var = numpy.var(store_pre_activation)

print("student prior induced mean {}".format(mean))
print("student prior induced var {}".format(var))








