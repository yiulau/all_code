import numpy
import pandas as pd

data = numpy.random.randn(5,2)
df = pd.DataFrame(data,columns=["a","b"])

print(df)