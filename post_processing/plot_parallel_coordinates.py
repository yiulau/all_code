from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import numpy,pandas

from sklearn.preprocessing import normalize

def parallel_coordinate_plot(sample_matrix,divergent_vec,normalize=False):
    num_features = sample_matrix.shape[1]
    column_names = ["Var {}".format(i) for i in range(num_features)]
    if normalize:
        normalized_x = numpy.zeros(sample_matrix.shape)
        for i in range(num_features):
            v = sample_matrix[:, i]
            normalized_x[:, i] = (v - v.min()) / (v.max() - v.min())
        df = pandas.DataFrame(normalized_x, columns=column_names)
    else:
        df = pandas.DataFrame(sample_matrix, columns=column_names)
    df["divergent"] = divergent_vec
    plt.figure()
    parallel_coordinates(df, "divergent")
    plt.show()
    return()



x = numpy.random.randn(50,4)
divergent = numpy.random.choice([0,1],size=50,replace=True)

parallel_coordinate_plot(x,divergent,True)

exit()
normalized_x = numpy.zeros((50,4))
for i in range(x.shape[1]):
    v = x[:,i]
    normalized_x[:,i] = (v - v.min())/(v.max()-v.min())

#print(normalized_x)

column_names = ["Var {}".format(i) for i in range(4)]

divergent = numpy.random.choice([0,1],size=50,replace=True)


#print(column_names)
df = pandas.DataFrame(x,columns=column_names)

df["divergent"] = divergent

df_normalized = pandas.DataFrame(normalized_x,columns=column_names)

df_normalized["divergent"] = divergent
#print(df.columns)
#exit()


plt.figure()
# print(df)
# df2 = df.columns.get_values()
# print(df2.tolist())

parallel_coordinates(df_normalized,"divergent")


plt.show()
print(df)