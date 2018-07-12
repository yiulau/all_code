import numpy
import pandas as pd

data = numpy.random.randn(5,2)

df = pd.DataFrame(data,columns=["a","b"],index=["rowa","rowb","rowc","rowd","rowe2"])

print(df)
df.to_csv("test.csv")

