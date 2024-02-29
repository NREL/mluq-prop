#!/usr/bin/env python

"""
Train Bayesian Neural Network model for scalar dissipation dataset.
================================

Two models are trained:
(a) Epistemic
(b) Epistemic+Aleatoric

The modeled features are:
- FC (* K-S equation)
- ChicF (* K-S equation)

TODO: add in low-fidelity model
TODO: use full dataset
TODO: add plotting of model performance
"""

import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from BNN.util import LRSchedule
from BNN.util.models import epi_bnn, bnn
from prettyPlot.plotting import pretty_legend, pretty_labels

__author__ = "Graham Pash"


def main(args):
    print("Num GPUs Available: ", len(tf.test.gpu_device_name()))

    # Load DNS data.
    data = np.load(args.datadir)
    Xtrain, Ytrain = data['Xtrain'], data['Ytrain']
    N, D_X = Xtrain.shape

    # Unpack input arguments
    np.random.seed(args.rng_key)  # for reproducibility
    SAVEDIR = "../models/leanKSE/"
    OPTIM = 'adam'

    D_H = args.hidden_dim
    batch_size = args.batch_size

    # Define callbacks.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=args.patience,
        restore_best_weights=True,
    )

    checkpoint_filepath = SAVEDIR + args.checkpath
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )

    lrschedule = tf.keras.callbacks.LearningRateScheduler(
        LRSchedule(initial_learning_rate=args.lrinit,
        final_learning_rate=args.lrfinal,
        epochs=args.max_iter,
        steps_per_epoch=N / batch_size,
        )
    )

    # Define model.
    lean = epi_bnn(
        train_size=batch_size,
        D_X=D_X,
        D_H=D_H,
        activation_fn=args.nonlin
    )
    
    # Tell everyone about the model you built.
    lean.summary()
    lean.compile(
        optimizer=OPTIM,
        loss='mse',
    )

    # Training.
    starttime = time()
    lean_history = lean.fit(
        Xtrain, Ytrain,
        epochs=args.max_iter,
        batch_size=batch_size,
        verbose=args.verbose,
        callbacks=[early_stop, model_checkpoint_callback, lrschedule],
    )
    print("Elapsed time:", time() - starttime)

    # Plot training history.
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    plt.plot(lean_history.history['loss'])
    pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Training History")
    plt.savefig(SAVEDIR + args.checkpath + args.filename + "_history.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Neural Network trained on KSE features")

    # ######## Setup ########
    parser.add_argument("--rng-key", type=int, default=0)
    parser.add_argument("--posterior", default="mvn", choices=["mvn", "independent"], help='use "mvn" or "independent"')
    parser.add_argument("--nonlin", default="sigmoid", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--datapath", default="/Users/gpash/Documents/uq-prop/data/leanKSE.npz")
    
    # ######## Output ########
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--filename", default="kse_dissipation")
    parser.add_argument("--checkpath", default="/Users/gpash/Documents/uq-prop/models/leanKSE/")

    # ######## Data Wrangling ########
    # TODO parser.add_argument("--normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=512)

    # ######## Inference ########
    parser.add_argument("--hidden-dim", default=5, type=int)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=5000)
    parser.add_argument("--lrinit", type=float, default=0.001)
    parser.add_argument("--lrfinal", type=float, default=0.0001)
        
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # ######## Run Script ########
    main(args)
