import matplotlib.pyplot as plt
import numpy as np

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)

# basic plot

data2 = np.random.randn(50,2)
plt.boxplot(data2)
plt.xticks([1, 2], ['mon', 'tue'])

#plt.show()
plt.savefig("test_simulate_outcome")