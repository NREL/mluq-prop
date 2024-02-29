#!/usr/bin/env python

"""
Train Bayesian Neural Network model for scalar dissipation dataset.
================================

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
import tensorflow as tf

from mluqprop.BNN.util import BNNCallbacks, save_history
from mluqprop.BNN.util.models import BNNHyperModel, neg_loglik
from mluqprop.BNN.util.dns import dns_data_loader, dns_partial_data_loader
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
    
    print("Training model with the following parameters:")
    pprint.pprint(simparams)

    # Load DNS data, define the data to be used for metrics logging.
    leandata = True if simparams.use_lean == "True" else False  # manually handle lean data case.
    if leandata:
        Xtrain, Ytrain = dns_partial_data_loader(simparams.data_fpath)
        Xmetric = Xtrain
        Ymetric = Ytrain
    else:
        Xtrain, Ytrain, Xval, Yval = dns_data_loader(simparams.data_fpath)
        Xmetric = Xval
        Ymetric = Yval
    
    # Get the data dimensions.
    N, DX = Xtrain.shape
    DY = Ytrain.shape[1]
    os.makedirs(simparams.savedir, exist_ok=True)
    
    # manually handle KL divergence rescaling
    KLWEIGHT = 1 / N if simparams.model_type == "variational" else 1.

    #############################
    # Define callbacks.
    #############################
    cbs = BNNCallbacks(simparams, Xmetric, Ymetric, N, args.verbose)
    cb = []

    warmup_best_ckpt = os.path.join(simparams.checkpath, "warmup/", "best")
    cb.append(cbs.best_ckpt_cb(warmup_best_ckpt))
    # # cb.append(cbs.lrschedule_cb())
    # if args.earlystop:
    #     cb.append(cbs.earlystop_cb())

    # if args.ckptepoch:
    #     cb.append(cbs.epoch_ckpt_cb())
    # else:
    #     metrics = cbs.metrics_cb()
    #     cb.append(metrics)

    #############################
    # Create the model.
    #############################
    abstractmodel = BNNHyperModel(dx=DX,
                                  dh=simparams.hidden_dim,
                                  dy=DY,
                                  nh=simparams.num_layers,
                                  kl_weight=0.,
                                  model_type=simparams.model_type,
                                  activation_fn=simparams.nonlin,
                                  posterior_model=simparams.posterior_model
                                )
    warmup = abstractmodel.build()

    # Tell everyone about the model you built.
    warmup.summary()
    warmup.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=simparams.learning_rate, global_clipnorm=simparams.grad_clip),
        optimizer=tf.keras.optimizers.Adam(learning_rate=simparams.learning_rate),
        loss=neg_loglik,
    )

    # Training.
    starttime = time()
    warmup_history = warmup.fit(
        Xtrain, Ytrain,
        epochs=simparams.epochs // 2,
        batch_size=simparams.batch_size,
        verbose=2,
        callbacks=cb
    )
    print("Elapsed time:", time() - starttime)
    
    # Plot training history.
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    plt.plot(warmup_history.history['loss'])
    pretty_labels(xlabel='Epoch', ylabel='ELBO', fontsize=18, title="Warmup Training History")
    plt.savefig(os.path.join(simparams.savedir, f"{simparams.filename}_warmup_history.pdf"))
    plt.close()

    #############################
    # Cleanup.
    #############################
    abstractmodel.kl_weight = KLWEIGHT
    model = abstractmodel.build()
    
    # Load best weights from the warmup.
    model.load_weights(warmup_best_ckpt)

    # Recompile the model with the correct KL weight.
    cb = []
    cb.append(cbs.best_ckpt_cb())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=simparams.learning_rate),
                  loss = neg_loglik)
    starttime = time()
    model_history = model.fit(
        Xtrain, Ytrain,
        epochs=simparams.epochs,
        batch_size=simparams.batch_size,
        verbose=2,
        callbacks=cb
    )
    print("Elapsed time:", time() - starttime)    
    

    # Plot training history.
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    plt.plot(model_history.history['loss'])
    pretty_labels(xlabel='Epoch', ylabel='ELBO', fontsize=18, title="Training History")
    plt.savefig(os.path.join(simparams.savedir, f"{simparams.filename}_history.pdf"))
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian Neural Network using DNS dataset.")

    parser.add_argument("--input", help="Input deck.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ckptepoch", action=argparse.BooleanOptionalAction, default=False, help="Checkpoint model instead of computing convergence statistics.")
    parser.add_argument("--earlystop", action=argparse.BooleanOptionalAction, default=False, help="Use early stopping callback.")
        
    # Optional arguments for batch submission, hyperparameter tuning, etc.
    parser.add_argument("--hidden_dim", type=int, default=-1, help="Number of hidden units.")
    parser.add_argument("--num_layers", type=int, default=-1, help="Number of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=-1, help="Initial learning rate.")
    parser.add_argument("--checkpath", type=str, default="", help="Path to save checkpoints.")
    parser.add_argument("--filename", type=str, default="", help="Filename for saved model.")


    # Parse the input arguments.
    args = parser.parse_args()
    if args.verbose:
        pprint.pprint(args)

    # Run the script.
    main(args)
