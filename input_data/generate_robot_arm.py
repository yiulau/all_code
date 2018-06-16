import numpy,math
def generate_robot_arms_data(number_of_ob):
    numpy.random.seed(1)
    pos_num = numpy.random.binomial(number_of_ob,0.5)
    x1 = numpy.zeros(number_of_ob)
    for i in range(number_of_ob):
        z = numpy.random.randn(1)
        if z > 0 :
            x1[i] =  numpy.random.uniform(0.453, 1.932, 1)
        else:
            x1[i] =         x1_neg = numpy.random.uniform(-1.932,-0.453,1)


    x2 = numpy.random.uniform(0.534,3.142,number_of_ob)
    y1 = 2 * numpy.cos(x1) + 1.3*numpy.cos(x1+x2) + numpy.random.randn(number_of_ob)*math.sqrt(0.05)
    y2 = 2 * numpy.sin(x1) + 1.3*numpy.sin(x1+x2) + numpy.random.randn(number_of_ob)*math.sqrt(0.05)

    input = numpy.column_stack((x1,x2))
    target = numpy.column_stack((y1,y2))

    out = {"input":input,"target":target}
    return(out)

def generate_cho_problem_data(number_of_ob):
    numpy.random.seed(1)
    x  = numpy.random.uniform(-1,1,number_of_ob)
    y = numpy.random.uniform(-1,1,number_of_ob)
    f = 0.3 + 1.2*x + 0.7*(y-0.2)*(y-0.2) + 2.3*y*(x+0.1)*(x+0.1)*(x+0.1)+numpy.random.randn(number_of_ob)*math.sqrt(0.1)

    X = numpy.column_stack((x,y))
    out = {"input":X,"target":f}

    return(out)

# generate_cho_problem_data(100)
#
# print(generate_robot_arms_data(100))