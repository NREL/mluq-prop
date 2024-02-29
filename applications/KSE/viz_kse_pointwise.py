#!/usr/bin/env python

"""
Plot a priori uncertainty estimates from BNN model.
================================

"""
import sys
import os
import argparse

import numpy as np
import pickle

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "util"))
import util.dictParser as inparser
import util.data as data
import util.postProc as postProc

# BNN model loader.
bnnpath = os.path.join(os.path.dirname(__file__), os.pardir)
bnnpath = os.path.join(bnnpath, os.pardir)
bnnpath = os.path.join(bnnpath, "BNN/util")
sys.path.append(bnnpath)
from models import load_net, compute_predictions

from prettyPlot.plotting import pretty_labels

def main(args):
    # Load in data.
    MODELDATA = "/Users/gpash/Documents/uq-prop/data/leanKSE.npz"
    KSDATA = "/Users/gpash/Documents/uq-prop/data/noSrc.npz"
    UPDATADIR = "/Users/gpash/Documents/uq-prop/data/prop_log/"
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

    Xtest = np.array([u.squeeze(), chicf.squeeze()]).T

    # Load in BNN model.
    MODELPATH = "/Users/gpash/Documents/uq-prop/models/leanKSE/"
    NUM_PREDICT = 250
    N_H = 2
    D_H = 5
    D_X = 2
    activation="sigmoid"
    model = load_net(MODELPATH,
        numHidden=N_H,
        numUnits=D_H,
        numInput=D_X,
        activation_fn=activation
    )

    preds, _, _ = compute_predictions(model, Xtest, num_samples=NUM_PREDICT)

    # Plot predictive uncertainty.
    plt.figure()
    plt.scatter(u, chicf, c=np.std(preds, axis=0), cmap='summer')
    cbar = plt.colorbar()
    pretty_labels(xlabel="Filtered Progress Variable",
        ylabel="Resolved Scalar Dissipation Rate",
        fontsize=18,
        title="Phase-Space Pointwise Uncertainty Estimate",
    )
    cbar.set_label("Estimated Standard Deviation")
    plt.show()

    # Predictive uncertainty on scaled phase-space.
    scaler = np.array([5, 15])

    preds, _, _ = compute_predictions(model, Xtest / scaler, num_samples=NUM_PREDICT)

    # Plot predictive uncertainty.
    plt.figure()
    plt.scatter(u / scaler[0], chicf / scaler [1], c=np.std(preds, axis=0), cmap='summer')
    cbar = plt.colorbar()
    pretty_labels(xlabel="Filtered Progress Variable",
        ylabel="Resolved Scalar Dissipation Rate",
        fontsize=18,
        title="Re-Scaled Phase-Space Pointwise Uncertainty Estimate",
    )
    cbar.set_label("Estimated Standard Deviation")
    plt.show()

    # A Posteriori Pointwise
    if args.posteriori:
        # Load in data from uncertainty propagation runs.
        # Initialize empty array and indexer.
        uu = None
        i = 0

        for fname in os.listdir(UPDATADIR):
            tmp = np.load(os.path.join(UPDATADIR, fname), allow_pickle=True)
            
            # On first call, allocate an array to store results.
            if uu is None:
                uu = np.empty((tmp['uu'].shape[0], tmp['uu'].shape[1], len(os.listdir(UPDATADIR))))
                tt = tmp['tt']
                kk = tmp['k']
            
            uu[:, :, i] = tmp['uu'].squeeze()
            i += 1

            # TODO: add in posteriori plotting


            if i >= args.max_runs:
                break
        
        # Compute gradients.
        u = uu.reshape((uu.shape[0]*uu.shape[1]*uu.shape[2], 1))
        N = len(u)

        # Compute resolved portion of scalar dissipation.
        v = np.fft.fft(u, axis=0)
        gradv = 1j * np.tile(kk.T, uu.shape[0]).T * v
        gradu = np.real(np.fft.ifft(gradv, axis=0))
        chicf = D * (gradu**2)

        # Randomly select X% of the data.
        idx = np.random.choice(range(N), size=int(args.downselect*N), replace=False)
        Xtest = np.array([u.squeeze(), chicf.squeeze()]).T
        Xtest = Xtest[idx, idx]

        # Plot predictive uncertainty.
        scaler = np.array([5, 15])

        preds, _, _ = compute_predictions(model, Xtest / scaler, num_samples=NUM_PREDICT)

        # Plot predictive uncertainty.
        plt.figure()
        plt.scatter(u / scaler[0], chicf / scaler[1], c=np.std(preds, axis=0), cmap='summer')
        cbar = plt.colorbar()
        pretty_labels(xlabel="Filtered Progress Variable",
            ylabel="Resolved Scalar Dissipation Rate",
            fontsize=18,
            title="Re-Scaled Phase-Space Pointwise Uncertainty Estimate",
        )
        cbar.set_label("Estimated Standard Deviation")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KS Equation Pointwise Uncertainty Estimates")

    parser.add_argument("--posteriori", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-predict", type=int, default=100)
    parser.add_argument("--max-runs", type=int, default=3)
       
    args = parser.parse_args()

    main(args)
