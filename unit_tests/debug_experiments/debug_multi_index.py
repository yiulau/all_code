import numpy

grid_shape = [2,4,5]
output_shape = [3,10]
store_grid_obj = numpy.empty(grid_shape, dtype=object)
output_grid = numpy.zeros(grid_shape+output_shape)
print(output_grid.shape)

it = numpy.nditer(store_grid_obj, flags=['multi_index', "refs_ok"])

while not it.finished:
    #store_grid_obj[it.multi_index] = numpy.random.randn(1)
    new_index = list(it.multi_index) + [...]
    #print(new_index)
    output_grid[new_index] = numpy.random.randn(*output_shape)
    it.iternext()


#print(store_grid_obj)
#print(output_grid)