#!/usr/bin/env python

###############################################################################
# 
# This script is intended to compare the performance of the
#  -- Bayesian Neural Network (BNN) models trained on the DNS data
#  -- Baseline Linear Relaxation Model (LRM)
#  -- Baseline deterministic Neural Network (DNN) model
# 
###############################################################################


"""
Postprocessing script for assessing trained Bayesian neural networks models.
"""

import os
import sys
import argparse
import pickle
import time

import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import numpy as np
import seaborn as sns

from mluqprop.BNN.util.metrics import compute_snr, compute_snr_flipout, vec_model_moments, conditionalAverage, weighted_norm
from mluqprop.BNN.util.models import load_model, compute_prob_predictions, compute_raw_epi_predictions, compute_epi_predictions_from_raw
from mluqprop.BNN.util.plotting import *
from mluqprop.BNN.util.dns import dns_data_loader, dns_partial_data_loader
from mluqprop.BNN.util.input_parser import parse_input_deck

# Malik's scaling codes.
from mluqprop import MLUQPROP_DATA_DIR
#sys.path.append("/Users/mhassana/Desktop/GitHub/S-mluq-prop_10d_jan12/data_10D/")
sys.path.append(MLUQPROP_DATA_DIR)
from scaling_upgraded import inv_scale_otpt_lrm, inv_scale_otpt, inv_scale_inpt, computeLRM_fromScaled, inv_scale_otpt_stdn, inv_scale_uncertainty


__author__ = "Graham Pash"

def main(args):
    # Parse the input deck.
    if args.input is None:
        raise NameError("No input deck provided. Please specify with: --input <input_deck>")
    simparams = parse_input_deck(args.input)
    
    if args.rescale:
        simparams.filename = simparams.filename + "_rescaled"
    
    # Load DNS data.
    simparams.use_lean = True if simparams.use_lean == "True" else False  # manually handle lean data flag.
    simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.
    if simparams.use_lean:
        Xtrain, Ytrain = dns_partial_data_loader(simparams.data_fpath)
        Xtest = Xtrain
        Ytest = Ytrain
    else:
        Xtrain, Ytrain, Xtest, Ytest = dns_data_loader(simparams.data_fpath)
        freq = 1
        Xtrain = Xtrain[::freq]
        Ytrain = Ytrain[::freq]
        Xtest = Xtest[::freq]
        Ytest = Ytest[::freq]
    N, D_X = Xtrain.shape
    DY = Ytrain.shape[1]
    
    # Load the model in.
    bnn_model = load_model(
        fpath=os.path.join(simparams.checkpath, "best/", "best"),
        D_X=D_X,
        D_H=simparams.hidden_dim,
        D_Y=DY,
        N_H=simparams.num_layers,
        kl_weight=1 / N if simparams.model_type == "variational" else 1.,
        model_type=simparams.model_type,
        activation_fn=simparams.nonlin,
        posterior_model=simparams.posterior_model,
        split=simparams.split
    )

    # Print the model summary.
    print(f"Model loaded from {simparams.checkpath}.")
    bnn_model.summary()


    PERCENTILES = [5, 95]
    print("Making epistemic uncertainty predictions...")
    print(f"Number of data points: {Xtest.shape[0]}")
    print(f"Number of predictions: {args.npredict}")
    start = time.time()
    
    #  we need the array of predictions to compute the epistemic uncertainty in physical space
    predicted, aleatory = compute_raw_epi_predictions(
            bnn_model, Xtest, num_samples=args.npredict
        )
    # compute the mean, uncertainties (in data space), and the percentiles (in data space)
    epipreds_mean, alestd, epistd, epi_ptile = compute_epi_predictions_from_raw(
        predicted, aleatory, ptiles=PERCENTILES
    )
    
    epipreds_mean = epipreds_mean[:, np.newaxis]
    alestd = alestd[:, np.newaxis]
    epistd = epistd[:, np.newaxis]
    print(f"Prediction took: {time.time() - start:.2f} seconds.")
    
    print("Generating full predictive envelope...")
    start = time.time()
    preds, preds_mean, preds_ptile = compute_prob_predictions(bnn_model, Xtest, num_samples=args.npredict, ptiles=PERCENTILES)
    preds_mean = preds_mean[:, np.newaxis]
    print(f"Prediction took: {time.time() - start:.2f} seconds.")
    
    # Save copies of the original predictions, in case rescaling is intermediately applied.
    epipreds_mean_orig = np.copy(epipreds_mean)
    alestd_orig = np.copy(alestd)
    epistd_orig = np.copy(epistd)
    epi_ptile_orig = np.copy(epi_ptile)
    
    preds_orig = np.copy(preds)
    preds_mean_orig = np.copy(preds_mean)
    preds_ptile_orig = np.copy(preds_ptile)
    Ytest_orig = np.copy(Ytest)

    # Optionally rescale all of the outputs.
    if args.rescale:
        epipreds_mean = inv_scale_otpt_lrm(epipreds_mean, Xtest)
        epistd = inv_scale_uncertainty(epistd)
        alestd = inv_scale_uncertainty(alestd)
        preds_mean = inv_scale_otpt_lrm(preds_mean, Xtest)
        Ytest = inv_scale_otpt_lrm(Ytest, Xtest)

    print("Making hex-scatter plot for the predictive mean.")
    PlotLogHexScatter(simparams, Ytest, np.array(epipreds_mean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BNN model performance on DNS data.")

    parser.add_argument("--input", help="Input deck.")
    parser.add_argument("--npredict", type=int, default=100)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True, help="Rescale data to physical space.")

    args = parser.parse_args()

    # ######## Run Script ########
    main(args)
