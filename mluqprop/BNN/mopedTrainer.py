#!/usr/bin/env python

"""
Train Bayesian Neural Network model for scalar dissipation dataset.

This script is for testing out custom training routines.
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
import numpy as np
import tensorflow as tf

from mluqprop.BNN.util import LRSchedule, Metrics, save_history
from mluqprop.BNN.util.models import neg_loglik, mlp, moped_bnn, NLL, BNNHyperModel
from mluqprop.BNN.util.dns import dns_data_loader, dns_partial_data_loader
from mluqprop.BNN.util.input_parser import parse_input_deck

from prettyPlot.plotting import pretty_labels

__author__ = "Graham Pash"

def main(args):
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    
    #############################
    # Parse the input deck.
    #############################
    if args.input is None:
        raise NameError("No input deck provided. Please specify with: --input <input_deck>")
    simparams = parse_input_deck(args.input)
    
    print("Training model with the following parameters:")
    pprint.pprint(simparams)

    # Load DNS data.
    leandata = True if simparams.use_lean == "True" else False  # manually handle lean data case.
    if leandata:
        Xtrain, Ytrain = dns_partial_data_loader(simparams.data_fpath)
    else:
        Xtrain, Ytrain, Xval, Yval = dns_data_loader(simparams.data_fpath)
    N, D_X = Xtrain.shape

    Ytrain = Ytrain

    # Unpack input arguments
    SAVEDIR = simparams.savedir
    N_H = simparams.num_layers
    D_H = simparams.hidden_dim
    batch_size = simparams.batch_size
    rescale = True if simparams.rescale == "True" else False  # manually handle KL divergence rescaling for DenseFlipout based models.

    #############################
    # Define callbacks.
    #############################
    cb = []
    
    if args.earlystop:
        # Early stopping callback.
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=simparams.patience,
            restore_best_weights=True,
        )
        cb.append(early_stop)

    # Model checkpoint callback for best model.
    best_ckpt_path = os.path.join(simparams.checkpath, "best/", "best")
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_ckpt_path,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    cb.append(checkpoint_best)

    # Learning rate scheduler callback.
    lrschedule = tf.keras.callbacks.LearningRateScheduler(
        LRSchedule(initial_learning_rate=simparams.lrinit,
            final_learning_rate=simparams.lrfinal,
            epochs=simparams.max_iter,
            steps_per_epoch=N / batch_size,
        )
    )
    cb.append(lrschedule)

    if args.ckptepoch:
        # Model checkpointing every so many epochs.
        checkpoint_epochs = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(simparams.checkpath, "epoch/", "epoch-{epoch:05d}.h5"),
            save_weights_only=True,
            monitor='loss',
            mode='min',
            period=simparams.skip
        )
        cb.append(checkpoint_epochs)
    else:
        # If not checkpointing every so many epochs, then compute statistics and log during training.
        # Load in the inducing points
        uips = np.load(simparams.uips_fpath)
        # Metrics callback.
        if leandata:
            metrics = Metrics(uips, Xtrain, Ytrain, simparams, verbose=args.verbose)
        else:    
            metrics = Metrics(uips, Xval, Yval, simparams, verbose=args.verbose)
        # cb.append(metrics)

    #############################
    # Do MOPED Initialization.
    #############################
    mlp_model = mlp(D_X=D_X, D_H=D_H, D_Y=1, N_H=N_H, activation_fn=simparams.nonlin)
    mlp_model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    
    # Model checkpoint callback for best model.
    mlp_ckpt_path = os.path.join(simparams.checkpath, "mlp/")
    mlp_ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=mlp_ckpt_path,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    
    print("Trianing MLP for MOPED initialization")
    starttime = time()
    model_history = mlp_model.fit(
        Xtrain, Ytrain,
        epochs=simparams.max_iter,
        batch_size=batch_size,
        verbose=2,
        callbacks=[mlp_ckpt]
    )
    print("Elapsed time:", time() - starttime)
    
    mlp_model.load_weights(mlp_ckpt_path).expect_partial()
    
    # LOSS
    # negloglik = lambda y, p_y: -p_y.log_prob(y)
    
    print("Reading MLP weights into BNN model.")
    moped_model = moped_bnn(mlp_model, delta=0.1, D_X=D_X, D_H=D_H, D_Y=1, N_H=2, activation_fn=simparams.nonlin, batch_size=batch_size)
    OPTIM = tf.keras.optimizers.Adam(learning_rate=simparams.lrinit, global_clipnorm=simparams.grad_clip)
    moped_model.summary()
    moped_model.compile(
        optimizer=OPTIM,
        loss=neg_loglik,
    )

    breakpoint()

    # Training.
    print("Training the MOPED Model.")
    tf.debugging.enable_check_numerics
    tf.config.run_functions_eagerly
    moped_model.run_eagerly = True
    
    starttime = time()
    model_history = moped_model.fit(
        Xtrain, Ytrain,
        epochs=simparams.max_iter,
        batch_size=batch_size,
        verbose=2,
        callbacks=cb
    )
    print("Elapsed time:", time() - starttime)

    #############################
    # Cleanup.
    #############################
    # Load best weights, save the model.
    moped_model.load_weights(best_ckpt_path)
    moped_model.save(os.path.join(SAVEDIR, simparams.filename))

    # Plot training history.
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    plt.plot(model_history.history['loss'])
    pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Training History")
    plt.savefig(os.path.join(SAVEDIR, f"{simparams.filename}_history.pdf"))
    plt.close()
    
    # Save history if necessary.
    if args.ckptepoch:
        with open(os.path.join(SAVEDIR, f"{simparams.filename}_history.pkl"), "wb") as f:
            pickle.dump(model_history.history, f)
    else:
        # TODO: might not need this anymore.
        save_history(model_history, metrics, os.path.join(SAVEDIR, f"{simparams.filename}_history.npz"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian Neural Network using DNS dataset.")

    parser.add_argument("--input", help="Input deck.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ckptepoch", action=argparse.BooleanOptionalAction, default=False, help="Checkpoint model instead of computing convergence statistics.")
    parser.add_argument("--earlystop", action=argparse.BooleanOptionalAction, default=False, help="Use early stopping callback.")
        
    # NOTE: batchsize, lrfinal, lrinit, num_layers, num_units defaults from SHERPA
    args = parser.parse_args()

    if args.verbose:
        pprint.pprint(args)

    # ######## Run Script ########
    main(args)
