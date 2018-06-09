import os, pickle
abs_address = os.environ["PYTHONPATH"] + "/input_data/subset_mnist.pkl"

data = pickle.load(open(abs_address, 'rb'))

inputdata = data["input"]

targetdata = data["target"]



print(inputdata.shape)

print(targetdata.shape)

print(targetdata)