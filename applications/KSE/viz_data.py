#!/usr/bin/env python

"""
Plot the 2D data distributions
(a) used to train BNN model
(b) outputted by the K-S equation with no source term
================================

"""
import sys
import os

import numpy as np
import pickle

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "util"))
from prettyPlot.plotting import pretty_labels, pretty_legend

# Load in data.
MODELDATA = "/Users/gpash/Documents/uq-prop/data/leanKSE.npz"
KSDATA = "/Users/gpash/Documents/uq-prop/data/noSrc.npz"
D = 1.

mdat = np.load(MODELDATA, allow_pickle=True)
ksdat = np.load(KSDATA, allow_pickle=True)

# Manipulate data to be the correct form.
xtrain = mdat['Xtrain']
xks = ksdat['uu']
u = xks.reshape((xks.shape[0]*xks.shape[1], 1))

# Compute resolved portion of scalar dissipation.
v = np.fft.fft(u, axis=0)
gradv = 1j * np.tile(ksdat['k'].T, xks.shape[0]).T * v
gradu = np.real(np.fft.ifft(gradv, axis=0))
chicf = D * (gradu**2)

# Co-plot the datasets.
plt.figure()
plt.scatter(u, chicf, color='gray', alpha=0.5, label="K-S Equation Data")
plt.scatter(xtrain[:, 0], xtrain[:, 1], color='blue', alpha=0.5, label="DNS Training Data")
pretty_legend()
pretty_labels(xlabel="Filtered Progress Variable",
    ylabel="Resolved Scalar Dissipation Rate",
    fontsize=18,
    title="Phase Space of Data Distributions",   
)

# Compute Z-score rescaling factors.
dnsmean = np.mean(xtrain, axis=0)
dnsstd = np.std(xtrain, axis=0)
ksvals = np.array([u, chicf]).squeeze().T
ksmean = np.mean(ksvals, axis=0)
ksstd = np.std(ksvals, axis=0)

# Hacky, but eyeball how the relative extremes need to be rescaled.
scaler = np.array([5, 15])
ksvals = ksvals / scaler

# Plot the rescaled data.
plt.figure()
plt.scatter(xtrain[:, 0], xtrain[:, 1], color='blue', alpha=0.5, label="DNS Training Data")
plt.scatter(ksvals[:, 0], ksvals[:, 1], color='gray', alpha=0.5, label="Scaled K-S Equation Data")
pretty_legend()
pretty_labels(xlabel="Filtered Progress Variable",
    ylabel="Resolved Scalar Dissipation Rate",
    fontsize=18,
    title="Phase Space of Re-Scaled Data Distributions",
    
)

# Show everyone what it looks like.
plt.show()
