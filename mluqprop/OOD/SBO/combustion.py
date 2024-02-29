import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sbo import soft_brownian_offset

sys.path.append("util")
from dataUtil import *

time_s = time.time()

X = np.load("../data/combustion2DToDownsampleSmall.npy")[:1000, :]
# X = np.load('../data/downSampledData_1000_best.npz')['data'][:,:]

# rescale the data
X_resc, minVal, maxVal = rescaleData(X)
# SBO on rescaled data
X_ood_resc = soft_brownian_offset(
    X_resc, d_min=2, d_off=2, n_samples=1000, softness=0
)
# Unrescale
X_ood = unrescaleData(X_ood_resc, minVal, maxVal)

time_e = time.time()
print(f"Elapsed time {time_e-time_s:.2f}s")

# Plot
markersize = 10
fig = plt.figure()
plt.plot(X[:, 0], X[:, 1], "o", color="k", markersize=markersize)
plt.plot(X_ood[:, 0], X_ood[:, 1], "x", color="r", markersize=markersize)
plt.show()
