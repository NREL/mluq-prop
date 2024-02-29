#!/usr/bin/env python

"""
Process single KS equation solution with custom title.
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
    DATAFILE = args.fname

    # Load in data.
    result = np.load(os.path.join(DATADIR, DATAFILE), allow_pickle=True)
    
    # Create dummy Sim dictionary to use same plotting routine.
    inpt = inparser.parseInputFile(args.inputfpath)
    Sim = data.simSetUp(inpt)

    Sim["plotTitle"] = args.title
    postProc.postProc_kse(result, Sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process single KS equation solution.")

    parser.add_argument("--datadir", default="/Users/gpash/Documents/uq-prop/data")
    parser.add_argument("--inputfpath", default="/Users/gpash/Documents/uq-prop/KSE/input_src")
    parser.add_argument("--fname", default="kse_meanBNN.npz")
    parser.add_argument("--title", default="Solution with BNN Mean as Forcing Function")
    # TODO: add in saving
    
    args = parser.parse_args()

    main(args)
