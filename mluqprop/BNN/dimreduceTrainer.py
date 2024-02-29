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

from mluqprop.BNN.util import BNNCallbacks
from mluqprop.BNN.util.models import BNNHyperModel, neg_loglik, load_model
from mluqprop.BNN.util.metrics import model2model
from mluqprop.BNN.util.dns import dns_data_loader, dns_partial_data_loader
from mluqprop.BNN.util.input_parser import parse_input_deck, check_user_input

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
    simparams.use_lean = True if simparams.use_lean == "True" else False  # manually handle lean data flag.
    simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.
    if simparams.use_lean:
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
                                  split=simparams.split,
                                  layer_mask=args.layer_mask
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

    # Compute difference to the reference model.
    refmodel = load_model(fpath=args.refmodel_path,
                            D_X=DX,
                            D_H=simparams.hidden_dim,
                            D_Y=DY,
                            N_H=simparams.num_layers,
                            kl_weight=KLWEIGHT,
                            model_type=simparams.model_type,
                            activation_fn=simparams.nonlin,
                            posterior_model=simparams.posterior_model,
                            split=simparams.split,
                            layer_mask=None
                        )

    wass, relmean, relale, relepi = model2model(model, refmodel, X=Xval, nepi=args.npredict)

    print(f"Wasserstein distance to reference model: {wass:.3e}")
    print(f"Relative norm of mean: {relmean:.3e}")
    print(f"Relative norm of aleatoric: {relale:.3e}")
    print(f"Relative norm of epistemic: {relepi:.3e}")
    
    # Save model training history.
    with open(os.path.join(simparams.savedir, f"{simparams.filename}_history.pkl"), "wb") as fid:
        pickle.dump(model_history.history, fid)
    
    # Write metrics to file.
    with open(os.path.join(simparams.savedir, f"{os.environ['SLURM_JOB_NAME']}.log"), "a") as fid:
        fid.write(f"{simparams.filename},{wass},{relmean},{relale},{relepi}\n")
    

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
    

    # Unique args for reduced model context.
    parser.add_argument("--layer_mask", nargs="+", type=int, help="Mask for probabilistic / deterministic layers.")
    parser.add_argument("--refmodel_path", type=str, help="Path to the reference BNN model.")
    parser.add_argument("--npredict", help="Number of predictions for model comparison.", type=int, default=50)
    
    args = parser.parse_args()

    if args.verbose:
        pprint.pprint(args)

    # ######## Run Script ########
    main(args)
