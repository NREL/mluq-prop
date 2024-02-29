#!/usr/bin/env python

"""
This script is for use when training multiple models, sweeping extrapolatory uncertainty datasets.
Because of this, more input arguments are required and some of the input deck args are overwritten.

The modeled features are:
- FC
- FCvar
- ChicF
- alpha
- beta
- gamma
- GradAlphaGradC
- GradBetaGradC
- GradGammaGradC
- FD
- GradTGradC
- FOmegaC
"""

import os
import argparse
import pickle
from time import time
import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mluqprop.BNN.util import BNNCallbacks, vec_moment_distance, wass1d, vec_model_moments, weighted_norm
from mluqprop.BNN.util.models import BNNHyperModel, neg_loglik
from mluqprop.BNN.util.dns import load_extrap_data
from mluqprop.BNN.util.input_parser import parse_input_deck, check_user_input

from prettyPlot.plotting import pretty_labels

__author__ = "Graham Pash"

def main(args):
    print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
    
    #############################
    # Parse the input deck.
    #############################
    if args.input is None:
        raise NameError("No input deck provided. Please specify with: --input <input_deck>")
    simparams = parse_input_deck(args.input)
    simparams = check_user_input(simparams, args)
    
    simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.
    
    print("Training model with the following parameters:")
    pprint.pprint(simparams)

    # Load DNS data.
    nextrap = -1 if args.nextrap == "sbo" else int(args.nextrap)
    Xtrain, Ytrain, Xval, Yval = load_extrap_data(simparams.data_fpath, args.extrap_data_fpath, nextrap)
    Xmetric = Xval
    Ymetric = Yval
    
    # Get the data dimensions.
    N, DX = Xtrain.shape
    DY = Ytrain.shape[1]
    os.makedirs(simparams.savedir, exist_ok=True)
    
    # manually handle KL divergence rescaling
    KLWEIGHT = 1 / N if simparams.model_type == "variational" else 1.
    KLWEIGHT = KLWEIGHT if simparams.klweight == -1 else simparams.klweight
    
    #############################
    # Define callbacks.
    #############################
    cbs = BNNCallbacks(simparams, Xmetric, Ymetric, N, args.verbose)
    cb = []

    cb.append(cbs.best_ckpt_cb())
    # cb.append(cbs.lrschedule_cb())
    if args.earlystop:
        cb.append(cbs.earlystop_cb())

    if args.ckptepoch:
        cb.append(cbs.epoch_ckpt_cb())

    if args.metrics:
        metrics = cbs.metrics_cb()
        cb.append(metrics)

    #############################
    # Create the model.
    #############################
    abstractmodel = BNNHyperModel(dx=DX,
                                  dh=simparams.hidden_dim,
                                  dy=DY,
                                  nh=simparams.num_layers,
                                  kl_weight=KLWEIGHT,
                                  model_type=simparams.model_type,
                                  activation_fn=simparams.nonlin,
                                  posterior_model=simparams.posterior_model,
                                  prior_model=simparams.prior_model,
                                  split=simparams.split
                                )
    model = abstractmodel.build()

     # Tell everyone about the model you built.
    model.summary()
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=simparams.learning_rate, global_clipnorm=simparams.grad_clip),
        optimizer=tf.keras.optimizers.Adam(learning_rate=simparams.learning_rate),
        loss=neg_loglik,
    )

    # Training.
    starttime = time()
    model_history = model.fit(
        Xtrain, Ytrain,
        epochs=simparams.epochs,
        batch_size=simparams.batch_size,
        verbose=2,
        callbacks=cb,
        validation_data=(Xval, Yval)
    )
    print("Elapsed time:", time() - starttime)

    #############################
    # Cleanup.
    #############################
    # Load best weights, save the model.
    model.load_weights(cbs.best_ckpt_path())

    # Load in the inducing points and compute the moment distances.
    uips = np.load(simparams.uips_fpath)
    extrap_uips = np.load(args.extrap_uips)
    
    # Moment Distance
    md, sd = vec_moment_distance(model, uips=uips, nalea=simparams.nalea, nepi=simparams.nepi)
    print(f"Relative norm of first moment with inducing points: {md:.3e}")
    print(f"Relative norm of second moment with inducing points: {sd:.3e}")
    
    # Weighted L2
    _, pmean, pepi, pale = vec_model_moments(model, uips["loc"], nalea=simparams.nalea, nepi=simparams.nepi)
    pmean = pmean[:, np.newaxis]
    pepi = pepi[:, np.newaxis]
    uipsmean = uips["mean"][:, np.newaxis]
    
    wrlmu = weighted_norm(uips["mean"]-pmean, uips["loc_prob"]) / weighted_norm(uips["mean"], uips["loc_prob"])
    wrlsig = weighted_norm(uips["aleat"]-pale, uips["loc_prob"]) / weighted_norm(uips["aleat"], uips["loc_prob"])
    print(f"Weighted Relative L2 Norm of First Moment with inducing points: {wrlmu:.3e}.")
    print(f"Weighted Relative L2 Norm of Second Moment with inducing points: {wrlsig:.3e}.")
    
    # Moment Distance (Extrapolatory)
    mdextrap, sdextrap = vec_moment_distance(model, uips=extrap_uips, nalea=simparams.nalea, nepi=simparams.nepi)
    print(f"Relative norm of first moment with inducing points (Extrapolatory): {mdextrap:.3e}")
    print(f"Relative norm of second moment with inducing points (Extrapolatory): {sdextrap:.3e}")
    
    # Weighted L2 (Extrapolatory)
    _, pmean, pepi, pale = vec_model_moments(model, extrap_uips["loc"], nalea=simparams.nalea, nepi=simparams.nepi)
    pmean = pmean[:, np.newaxis]
    
    # extrapwrlmu = weighted_norm(extrap_uips["mean"]-pmean, extrap_uips["loc_prob"]) / weighted_norm(extrap_uips["mean"], extrap_uips["loc_prob"])
    # extrapwrlsig = weighted_norm(extrap_uips["aleat"]-pale, extrap_uips["loc_prob"]) / weighted_norm(extrap_uips["aleat"], extrap_uips["loc_prob"])
    extrapwrlmu = 0   # 10D doesn't have location probability.
    extrapwrlsig = 0  # 10D doesn't have location probability.
    # print(f"Weighted Relative L2 Norm of First Moment with inducing points (Extrapolatory): {extrapwrlmu:.3e}.")
    # print(f"Weighted Relative L2 Norm of Second Moment with inducing points (Extrapolatory): {extrapwrlsig:.3e}.")
    
    # Wasserstein Distance.
    wass = wass1d(Yval, model(Xval).sample().numpy())
           
    # Plot training history.
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    plt.plot(model_history.history['loss'])
    pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Training History")
    plt.savefig(os.path.join(simparams.savedir, f"{simparams.filename}_history.pdf"))
    plt.close()
    
    # Save model training history.
    with open(os.path.join(simparams.savedir, f"{simparams.filename}_history.pkl"), "wb") as fid:
        pickle.dump(model_history.history, fid)
    
    # Write metrics to file.
    with open(os.path.join(simparams.savedir, f"{os.environ['SLURM_JOB_NAME']}.log"), "a") as fid:
        fid.write(f"{simparams.filename},{md},{sd},{wrlmu},{wrlsig},{mdextrap},{sdextrap},{extrapwrlmu},{extrapwrlsig},{wass}\n")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian Neural Network using DNS dataset.")

    parser.add_argument("--input", help="Input deck.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ckptepoch", action=argparse.BooleanOptionalAction, default=True, help="Checkpoint model instead of computing convergence statistics.")
    parser.add_argument("--earlystop", action=argparse.BooleanOptionalAction, default=False, help="Use early stopping callback.")
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True, help="Use metrics callback.")
    
    # Optional arguments for batch submission, hyperparameter tuning, etc.
    parser.add_argument("--hidden_dim", type=int, default=-1, help="Number of hidden units.")
    parser.add_argument("--num_layers", type=int, default=-1, help="Number of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=-1, help="Initial learning rate.")
    parser.add_argument("--checkpath", type=str, default="", help="Path to save checkpoints.")
    parser.add_argument("--filename", type=str, default="", help="Filename for saved model.")
   
    # Extrapolatory uncertainty data type and paths.
    parser.add_argument("--extraptype", default="nf", choices=["nf", "sbo"], help="Extrapolation type. Either 'nf' for normalizing flow, or 'sbo' for Soft Brownian Offset.")
    parser.add_argument("--extrap_data_fpath", help="Path to the data file with extrapolation data.")
    parser.add_argument("--extrap_uips", help="Path to the inducing points for the extrapolation data.")
    parser.add_argument("--nextrap", help="Number of extrapolatroy data points to use. Only for NF.", type=int, default=-1)

    args = parser.parse_args()

    if args.verbose:
        pprint.pprint(args)

    # ######## Run Script ########
    main(args)
