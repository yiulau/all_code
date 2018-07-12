import matplotlib.pyplot as plt
import numpy as np

a = np.random.random((5, 5))
plt.imshow(a,  aspect="auto",
           cmap=plt.get_cmap("spring"))
clb = plt.colorbar()
clb.ax.set_title('log scale')
#plt.show()

plt.savefig("test_heatmap.png")
