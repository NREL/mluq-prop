#!/usr/bin/env python

"""
Continue trianing of a partially trained Bayesian neural network model.

TODO: This script is not yet complete.
"""

import argparse
import os
import pickle
import pprint
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mluqprop.BNN.util import LRSchedule, Metrics, load_model, save_history
from mluqprop.BNN.util.dns import dns_data_loader
from mluqprop.BNN.util.input_parser import parse_input_deck
from prettyPlot.plotting import pretty_labels

__author__ = "Graham Pash"


def main(args):
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    #############################
    # Parse the input deck.
    #############################
    if args.input is None:
        raise NameError(
            "No input deck provided. Please specify with: --input <input_deck>"
        )
    simparams = parse_input_deck(args.input)

    print("Continuing to train model with the following parameters:")
    pprint.pprint(simparams)

    # Load DNS data.
    Xtrain, Ytrain, Xval, Yval = dns_data_loader(simparams.data_fpath)
    N, D_X = Xtrain.shape

    # Unpack input arguments
    SAVEDIR = simparams.savedir
    batch_size = simparams.batch_size

    #############################
    # Define callbacks.
    #############################
    cb = []

    if args.earlystop:
        # Early stopping callback.
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=simparams.patience,
            restore_best_weights=True,
        )
        cb.append(early_stop)

    # Model checkpoint callback for best model.
    best_ckpt_path = os.path.join(simparams.checkpath, "best/", "best")
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_ckpt_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    cb.append(checkpoint_best)

    # Learning rate scheduler callback.
    lrschedule = tf.keras.callbacks.LearningRateScheduler(
        LRSchedule(
            initial_learning_rate=simparams.lrinit,
            final_learning_rate=simparams.lrfinal,
            epochs=simparams.max_iter,
            steps_per_epoch=N / batch_size,
        )
    )
    cb.append(lrschedule)

    if args.ckptepoch:
        # Model checkpointing every so many epochs.
        checkpoint_epochs = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                simparams.checkpath, "epoch/", "epoch-{epoch:05d}.h5"
            ),
            save_weights_only=True,
            monitor="loss",
            mode="min",
            period=simparams.skip,
        )
        cb.append(checkpoint_epochs)
    else:
        # If not checkpointing every so many epochs, then compute statistics and log during training.
        # Load in the inducing points
        uips = np.load(simparams.uips_fpath)
        # Metrics callback.
        metrics = Metrics(uips, Xval, Yval, simparams, verbose=args.verbose)
        cb.append(metrics)

    # Load the model back in.
    model = load_model(
        fpath=os.path.join(simparams.modelpath, "best/", "best"),
        D_X=D_X,
        D_H=simparams.hidden_dim,
        D_Y=1,
        N_H=simparams.num_layers,
        train_size=simparams.batch_size,
        model_type=simparams.model_type,
        activation_fn=simparams.nonlin,
        posterior_model=simparams.posterior_model,
    )

    model.summary()

    if args.recompile_model:
        # If changing the optimizer, recompile.
        OPTIM = tf.keras.optimizers.Adam(
            learning_rate=simparams.lrinit, global_clipnorm=simparams.clipnorm
        )
        model.compile(optimizer=OPTIM, loss="mse")

    # Continue training.
    starttime = time()
    model_history = model.fit(
        Xtrain,
        Ytrain,
        epochs=simparams.max_iter,
        batch_size=batch_size,
        verbose=2,
        callbacks=cb,
    )
    print("Elapsed time:", time() - starttime)

    #############################
    # Cleanup.
    #############################
    # Load best weights, save the model.
    model.load_weights(best_ckpt_path)
    model.save(os.path.join(SAVEDIR, simparams.filename))
    model.save_weights(os.path.join(SAVEDIR, f"{simparams.filename}_weights"))

    # Plot training history.
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    plt.plot(model_history.history["loss"])
    pretty_labels(
        xlabel="Iteration",
        ylabel="ELBO",
        fontsize=18,
        title="Training History",
    )
    plt.savefig(
        os.path.join(SAVEDIR, f"{simparams.filename}_continued_history.pdf")
    )
    plt.close()

    # Save history if necessary.
    if args.ckptepoch:
        with open(
            os.path.join(SAVEDIR, f"{simparams.filename}_history.pkl"), "wb"
        ) as f:
            pickle.dump(model_history.history, f)
    else:
        # TODO: might not need this anymore.
        save_history(
            model_history,
            metrics,
            os.path.join(
                SAVEDIR, f"{simparams.filename}_continued_history.npz"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continue training of saved model."
    )

    parser.add_argument("--input", help="Input deck.")
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save-history", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--recompile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ckptepoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint model instead of computing convergence statistics.",
    )
    parser.add_argument(
        "--earlystop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use early stopping callback.",
    )

    args = parser.parse_args()

    print(args)

    # Run script.
    main(args)
