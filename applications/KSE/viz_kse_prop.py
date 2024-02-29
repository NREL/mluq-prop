#!/usr/bin/env python

"""
Process uncertainty propagation results for the KS equation simulation.
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
from prettyPlot.plotting import pretty_labels


def main(args):
    DATADIR = args.datadir

    # Initialize empty array and indexer.
    uu = None
    i = 0

    for fname in os.listdir(DATADIR):
        tmp = np.load(os.path.join(DATADIR, fname), allow_pickle=True)
        
        # On first call, allocate an array to store results.
        if uu is None:
            uu = np.empty((tmp['uu'].shape[0], tmp['uu'].shape[1], len(os.listdir(DATADIR))))
            tt = tmp['tt']
        
        uu[:, :, i] = tmp['uu'].squeeze()
        i += 1

    uu_mean = np.mean(uu, axis=2)[..., np.newaxis]
    uu_std = np.std(uu, axis=2)[..., np.newaxis]

    # Create dummy Sim dictionary to use same plotting routine.
    inpt = inparser.parseInputFile(args.inputfpath)
    Sim = data.simSetUp(inpt)

    meanResult = {"uu":uu_mean, "tt":tt}
    stdResult = {"uu":uu_std, "tt":tt}

    Sim["plotTitle"] = "Mean Realization"
    postProc.postProc(meanResult, Sim)
    Sim["plotTitle"] = "Pointwise Uncertainty"
    Sim["clabel"] = "Standard Deviation"
    postProc.postProc(stdResult, Sim)

    if args.show_realizations:
        for i in np.random.choice(range(uu.shape[2]), size=10):
            postProc.postProc({'uu':uu[:, :, i][..., np.newaxis]}, Sim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KS equation uncertainty propagation.")

    parser.add_argument("--datadir", default="/Users/gpash/Documents/uq-prop/data/prop_log")
    parser.add_argument("--inputfpath", default="/Users/gpash/Documents/uq-prop/KSE/input_src")
    parser.add_argument("--show_realizations", action=argparse.BooleanOptionalAction, default=False)
    # TODO: add in saving
    
    args = parser.parse_args()

    main(args)
