#!/usr/bin/env python

"""
Comparison of Bayesian Neural Networks
================================

This example is meant to compare a Bayesian Neural Network that models:
(a) Epistemic Uncertainty ONLY
(b) Epistemic + Aleatoric Uncertainty

Additional model(s):
(c) Deterministic Multilayer Perceptron
(d) Gaussian Process with RBF kernel
"""

import argparse
from time import time

from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import os

from mluqprop.BNN.util import compute_prob_predictions, compute_predictions
from mluqprop.BNN.util.models import epi_bnn, bnn, mlp, flipout_bnn
from mluqprop.BNN.util.models import isotropic_posterior, mvn_posterior, neg_loglik
from mluqprop.BNN.util.input_parser import parse_input_deck
from prettyPlot.plotting import pretty_legend, pretty_labels

import gpflow  # Could also use sci-kit learn

__author__ = "Graham Pash"

def main(args):
    #############################
    # Parse the input deck.
    #############################
    if args.input is None:
        raise NameError("No input deck provided. Please specify with: --input <input_deck>")
    simparams = parse_input_deck(args.input)
    
    np.random.seed(simparams.seed)  # for reproducibility
    SAVEDIR = args.savedir
    OPTIM = simparams.optimizer
    xlim_train = [-1, 1]
    xlim_test = [-4, 4]

    # Unpack input arguments.
    Ntrain = simparams.num_train_data
    Ntest = simparams.num_test_data
    D_X = 1
    D_H = simparams.hidden_dim
    percentiles = [50. - simparams.percentiles/2., 50. + simparams.percentiles/2.]

    if simparams.posterior_model == 'mvn':
        POSTMODEL = mvn_posterior
    elif simparams.posterior_model == 'independent':
        POSTMODEL = isotropic_posterior
    else:
        raise ValueError("Invalid posterior type.")

    # Generate data.
    ydata = lambda x, n: np.power(x, 3) + simparams.noise_level*(1.5 + x)*np.random.randn(n)

    Xtrain = np.linspace(xlim_train[0], xlim_train[1], Ntrain)[:, np.newaxis]
    Ytrain = ydata(Xtrain.squeeze(), Ntrain)[:, np.newaxis]

    Xtest = np.linspace(xlim_test[0], xlim_test[1], Ntest)[:, np.newaxis]
    Ytest = ydata(Xtest.squeeze(), Ntest)[:, np.newaxis]


    SHIFT = 0
    Ytrain = Ytrain + SHIFT
    Ytest = Ytest + SHIFT

    # Plot data.
    fig, ax = plt.subplots()
    plt.scatter(Xtest, Ytest, alpha=0.5, color='gray', label='Test Data')
    plt.plot(Xtrain, Ytrain, 'kx', label='Training Data')

    # If training a deterministic network, do that first.
    if args.do_mlp:
        # Define callbacks.
        checkpoint_filepath_mlp = '/tmp/mlp'
        model_checkpoint_callback_mlp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_mlp,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True,
        )
        print("Training Multilayer Perceptron")
        mlp_model = mlp(D_X=1, D_H=D_H, activation_fn=simparams.nonlin)
        mlp_model.summary()
        mlp_model.compile(
            optimizer=OPTIM,
            loss='mse',
        )
        mlp_history = mlp_model.fit(
            Xtrain, Ytrain,
            epochs=simparams.max_iter,
            verbose=args.verbose,
            callbacks=[model_checkpoint_callback_mlp],
        )

        # Restore best model.
        mlp_model.load_weights(checkpoint_filepath_mlp)
        mlp_pred = mlp_model(Xtest)

        # Add MLP prediction to plot.
        plt.plot(Xtest, mlp_pred, 'k', linewidth=2.0, label="Multilayer Perceptron")

    #############################
    # Epistemic Uncertainty ONLY
    #############################
    # Define callbacks.
    checkpoint_filepath = '/tmp/epi'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    epi_model = epi_bnn(kl_weight=1/Ntrain, D_X=D_X, D_H=D_H, posterior_fn=POSTMODEL, activation_fn=simparams.nonlin)
    epi_model.compile(
        optimizer=OPTIM,
        loss='mse',
    )
    print("Training Epistemic BNN.")
    checkpoint_filepath_epi = '/tmp/checkpoint_epi'
    model_checkpoint_callback_epi = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_epi,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    epi_model.summary()
    starttime = time()
    epi_history = epi_model.fit(
        Xtrain, Ytrain,
        epochs=simparams.max_iter,
        verbose=args.verbose,
        callbacks=[model_checkpoint_callback_epi],
    )
    print(f"Elapsed epistemic training time: {time() - starttime:.2f}s")

    # Load best weights, make predictions. 
    epi_model.load_weights(checkpoint_filepath_epi).expect_partial()
    epi_preds, epi_mean, epi_ptiles = compute_predictions(epi_model, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

    # Add epistemic model to plot.
    ax.plot(Xtest, epi_mean, "blue", ls="solid", lw=2.0, label="Epistemic Only")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), epi_ptiles[0, :], epi_ptiles[1, :], color="lightblue", alpha=0.5
    )

    #############################
    # Epistemic & Aleatoric
    #############################
    # Define callbacks.
    checkpoint_filepath_bnn = '/tmp/full'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_bnn,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    bnn_model = bnn(kl_weight=1/Ntrain, D_X=D_X, D_H=D_H, posterior_fn=POSTMODEL, activation_fn=simparams.nonlin)
    bnn_model.compile(
        optimizer=OPTIM,
        loss=neg_loglik,
    )
    print("Training Epistemic+Aleatoric BNN.")
    bnn_model.summary()
    starttime = time()
    bnn_history = bnn_model.fit(
        Xtrain, Ytrain,
        epochs=simparams.max_iter,
        verbose=args.verbose,
        callbacks=[model_checkpoint_callback],
    )
    print(f"Elapsed full model training time: {time() - starttime:.2f}s")

    # Load best weights, make predictions. 
    bnn_model.load_weights(checkpoint_filepath_bnn).expect_partial()
    bnn_preds, bnn_mean, bnn_ptiles = compute_prob_predictions(bnn_model, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

    # Add full Bayesian model to plot.
    ax.plot(Xtest, bnn_mean, "orangered", ls="solid", lw=2.0, label="Epistemic+Aleatoric")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), bnn_ptiles[0, :], bnn_ptiles[1, :], color="lightcoral", alpha=0.5
    )
    
    #############################
    # Flipout Model
    #############################
    checkpoint_filepath_flipout = '/tmp/flipout'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_flipout,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    flipout_model = flipout_bnn(kl_weight=1/Ntrain, D_X=D_X, D_H=D_H, activation_fn=simparams.nonlin)
    flipout_model.compile(
        optimizer=OPTIM,
        loss=neg_loglik,
    )
    print("Training Flipout BNN.")
    flipout_model.summary()
    starttime = time()
    flipout_history = flipout_model.fit(
        Xtrain, Ytrain,
        epochs=simparams.max_iter,
        verbose=args.verbose,
        callbacks=[model_checkpoint_callback],
    )
    print("Elapsed flipout model training time:", time() - starttime)

    # Load best weights, make predictions. 
    flipout_model.load_weights(checkpoint_filepath_flipout).expect_partial()
    flipout_preds, flipout_mean, flipout_ptiles = compute_prob_predictions(bnn_model, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

    # Add full Bayesian model to plot.
    ax.plot(Xtest, flipout_mean, "purple", ls="solid", lw=2.0, label="Flipout")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), flipout_ptiles[0, :], flipout_ptiles[1, :], color="darkorchid", alpha=0.5
    )

    #############################
    # Gaussian Process Regression
    #############################
    if args.do_gp:
        k = gpflow.kernels.RBF()
        gpr = gpflow.models.GPR(data=(Xtrain, Ytrain), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        
        # Train GP.
        opt_logs = opt.minimize(gpr.training_loss, gpr.trainable_variables, options=dict(maxiter=simparams.max_iter))

        # Make predictions.
        gp_mean, gp_sd = gpr.predict_f(Xtest)
        gp_mean = gp_mean.numpy()
        gp_sd = gp_sd.numpy()
        ax.plot(Xtest, gp_mean, "gray", ls="solid", lw=2.0, label="Gaussian Process")
        z = NormalDist().inv_cdf((1 + simparams.percentiles/100.) / 2.)
        ax.fill_between(
            Xtest.squeeze(),
            (gp_mean - z * gp_sd).squeeze(),
            (gp_mean + z * gp_sd).squeeze(),
            color="lightslategray",
            alpha=0.3,
        )


    #############################
    # Finalize Output
    #############################
    pretty_labels(xlabel='x', ylabel='y', fontsize=18, title=f'Mean Predictions with {simparams.percentiles:d}% Confidence Interval')
    pretty_legend()
    plt.ylim(-5, 5+SHIFT)
    plt.savefig(os.path.join(SAVEDIR, f"shifted_mini_{simparams.filename}.pdf"))
    plt.close()

    # Compare DenseFlipout and DenseVariational models.
    fig, ax = plt.subplots()
    plt.scatter(Xtest, Ytest, alpha=0.5, color='gray', label='Test Data')
    plt.plot(Xtrain, Ytrain, 'kx', label='Training Data')
    
    # Add Flipout model to plot.
    ax.plot(Xtest, flipout_mean, "purple", ls="solid", lw=2.0, label="Flipout")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), flipout_ptiles[0, :], flipout_ptiles[1, :], color="darkorchid", alpha=0.5
    )
    
    # Add full Bayesian model to plot.
    ax.plot(Xtest, bnn_mean, "orangered", ls="solid", lw=2.0, label="Epistemic+Aleatoric")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), bnn_ptiles[0, :], bnn_ptiles[1, :], color="lightcoral", alpha=0.5
    )
    
    pretty_labels(xlabel='x', ylabel='y', fontsize=18, title=f'Mean Predictions with {simparams.percentiles:d}% Confidence Interval')
    pretty_legend()
    plt.ylim(-5, 5+SHIFT)
    plt.savefig(os.path.join(SAVEDIR, f"shifted_mini_{simparams.filename}_bnns.pdf"))
    plt.close()

    #############################
    # Plot Individual Realizations
    #############################
    Nplots = simparams.num_plots
    
    # Epistemic model.
    fig, ax = plt.subplots()
    for _ in range(Nplots):
        pred = epi_model(Xtest).numpy()
        plt.plot(Xtest, pred, 'k', lw=0.5)
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), epi_ptiles[0, :], epi_ptiles[1, :], color="gray", alpha=0.5
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Epistemic Model Realizations')
    plt.savefig(os.path.join(SAVEDIR, "epi_realizations.pdf"))

    # Epistemic & Aleatoric model.
    fig, ax = plt.subplots()
    for _ in range(Nplots):
        pred = bnn_model(Xtest).sample()
        plt.plot(Xtest, pred, 'k', lw=0.5)
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), bnn_ptiles[0, :], bnn_ptiles[1, :], color="gray", alpha=0.5
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Full Model Realizations')
    plt.savefig(os.path.join(SAVEDIR, "full_realizations.pdf"))

    # Gaussian Process
    if args.do_gp:
        gp_samples = gpr.predict_f_samples(Xtest, Nplots)
        fig, ax = plt.subplots()
        for i in range(Nplots):
            pred = gp_samples[i, :, :].numpy()
            plt.plot(Xtest, pred, 'k', lw=0.5)
        ax.fill_between(
            Xtest.squeeze(),
            (gp_mean - z * gp_sd).squeeze(),
            (gp_mean + z * gp_sd).squeeze(),
            color="gray",
            alpha=0.5,
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gaussian Process Realizations')
        plt.savefig(os.path.join(SAVEDIR, "gp_realizations.pdf"))

    # Plot Training histories
    if args.save_history:
        # Epistemic model.
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(epi_history.history['loss'])
        pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Epistemic Training History")
        plt.savefig(os.path.join(SAVEDIR, "epi_history.pdf"))
        plt.close()

        # Full Bayesian model.
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(bnn_history.history['loss'])
        pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Epistemic+Aleatoric Training History")
        plt.savefig(os.path.join(SAVEDIR, "full_history.pdf"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration of Bayesian Neural Network uncertainties on 1D dataset")

    # Set up input arguments.
    parser.add_argument("--input", help="Input deck.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--savedir", default="Figures/uncertainties/")
    parser.add_argument("--do-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do-gp", action=argparse.BooleanOptionalAction, default=True)
        
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # ######## Run Script ########
    main(args)
