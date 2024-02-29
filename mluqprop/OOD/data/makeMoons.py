import numpy as np
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=100000, noise=0.08)
dim = 2

np.save("moons.npy", X)
