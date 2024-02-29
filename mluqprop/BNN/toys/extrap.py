#!/usr/bin/env python

"""
Testing different methods of incorporating prior knowledge into BNN models
================================

This example is meant to compare different methods for setting priors on the network
NOTE: priors is potentially a misnomer, we are in fact concerned with the
        incorporation of low-fidelity data within the model.

(a) Warm-start with low-fidelity model
(b) Retention of low-fidelity data
(c) MOPED prior/initialization via DenseFlipout
(d) Empirical Bayes via trainable prior
---------------------------------
(TODO) GP prior papers
(TODO) fBNN

TODO: add optional data normalization
"""

import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np

from sbo import soft_brownian_offset

import tensorflow as tf

import os
import sys

from mluqprop.BNN.util import compute_prob_predictions
from mluqprop.BNN.util.models import bnn, isotropic_posterior, mvn_posterior, neg_loglik
from mluqprop.BNN.util.models import mlp, moped_bnn, BNNHyperModel
from mluqprop.BNN.util.input_parser import parse_input_deck
from prettyPlot.plotting import pretty_legend, pretty_labels

__author__ = "Graham Pash"

def main(args):
    #############################
    # Parse the input deck.
    #############################
    if args.input is None:
        raise NameError("No input deck provided. Please specify with: --input <input_deck>")
    simparams = parse_input_deck(args.input)
        
    #############################
    # Setup
    #############################
    np.random.seed(simparams.seed)  # for reproducibility
    SAVEDIR = args.savedir
    OPTIM = simparams.optimizer
    xlim_train = [-1, 1]
    xlim_synth = [-2, 2]
    xlim_test = [-4, 4]

    # Unpack input arguments.
    Ntrain = simparams.num_train_data
    Ntest = simparams.num_test_data
    Nsynth = simparams.num_synth_data
    Ntot = Ntrain + Nsynth
    D_X = 1
    D_Y = 1
    D_H = simparams.hidden_dim
    percentiles = [50. - simparams.percentiles/2., 50. + simparams.percentiles/2.]

    # Generate data.
    ysynth = lambda x, n: np.power(x, 2) + simparams.synth_noise*np.random.randn(n)
    ydata = lambda x, n: np.power(x, 3) + simparams.noise_level*(1.5 + x)*np.random.randn(n)

    Xsynth = np.linspace(xlim_synth[0], xlim_synth[1], Ntrain)[:, np.newaxis]
    Ysynth = ysynth(Xsynth.squeeze(), Ntrain)[:, np.newaxis]

    Xtrain = np.linspace(xlim_train[0], xlim_train[1], Ntrain)[:, np.newaxis]
    Ytrain = ydata(Xtrain.squeeze(), Ntrain)[:, np.newaxis]

    Xtest = np.linspace(xlim_test[0], xlim_test[1], Ntest)[:, np.newaxis]
    Ytest = ydata(Xtest.squeeze(), Ntest)[:, np.newaxis]

    Xood = soft_brownian_offset(Xtrain, d_min=simparams.min_offset, d_off=0.5, n_samples=Nsynth, softness=1)
    Yood = ysynth(Xood.squeeze(), Nsynth)[:, np.newaxis]
    Xtot = np.concatenate((Xtrain, Xood))
    Ytot = np.concatenate((Ytrain, Yood))

    # Plot data.
    fig, ax = plt.subplots()
    plt.scatter(Xtest, Ytest, alpha=0.5, color='gray', label='True Data')
    plt.plot(Xtrain, Ytrain, 'kx', label='High-Fidelity Training Data')
    plt.plot(Xood, Yood, 'rx', label='Low-Fidelity Training Data')

    #############################
    # Warm-Start Model Training
    #############################

    # Define callbacks.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5000,
        restore_best_weights=True,
    )

    checkpoint_filepath_warmstart = '/tmp/checkpoint_warmstart'
    model_checkpoint_callback_warmstart = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_warmstart,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )

    # Build, train Warm-start.
    print("Training BNN with warm-start.")
    abstractmodel = BNNHyperModel(
        dx=D_X,
        dh=simparams.hidden_dim,
        dy=D_Y,
        nh=simparams.num_layers,
        kl_weight=1 / Nsynth,
        model_type="variational",
        activation_fn=simparams.nonlin,
        posterior_model="mvn",
        prior_model="isotropic",
    )
    warm_start = abstractmodel.build()
    
    warm_start.summary()  # output model summary to console
    warm_start.compile(
        optimizer=OPTIM,
        loss=neg_loglik,
    )
    starttime = time()
    warm_history = warm_start.fit(
        Xsynth, Ysynth,
        epochs=simparams.max_iter,
        verbose=args.verbose,
        callbacks=[early_stop, model_checkpoint_callback_warmstart],
    )
    print(f"Elapsed warmup time {time() - starttime:.2f}s")

    # Load best weights, make predictions. 
    warm_start.load_weights(checkpoint_filepath_warmstart)
    warmup_preds, warmup_mean, warmup_ptiles = compute_prob_predictions(warm_start, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

    # Add warmup model predictions to plot.
    ax.plot(Xtest, warmup_mean, "blue", ls="solid", lw=2.0, label="Trained Prior")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), warmup_ptiles[0, :], warmup_ptiles[1, :], color="lightblue", alpha=0.5
    )
    
    checkpoint_filepath_HF = '/tmp/checkpoint_warmstart_cont'
    model_checkpoint_callback_HF = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_HF,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    # Continue training on high-fidelity data.
    print("Continuing training with high-fidelity data.")
    starttime = time()
    full_history = warm_start.fit(
        Xtrain, Ytrain,
        epochs=simparams.max_iter,
        verbose=args.verbose,
        callbacks=[model_checkpoint_callback_HF],
    )
    print(f"Warm start elapsed time {time() - starttime:.2f}s")

    # Load best weights, make predictions. 
    warm_start.load_weights(checkpoint_filepath_HF)
    post_preds, post_mean, post_ptiles = compute_prob_predictions(warm_start, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

    # Add retrained predictions to plot.
    ax.plot(Xtest, post_mean, "orangered", ls="solid", lw=2.0, label="Warm-Start Posterior")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), post_ptiles[0, :], post_ptiles[1, :], color="lightcoral", alpha=0.5
    )

    #############################
    # Training with Combined Data
    #############################
    print("Training model with combined datasets.")
    abstractmodel = BNNHyperModel(
        dx=D_X,
        dh=simparams.hidden_dim,
        dy=D_Y,
        nh=simparams.num_layers,
        kl_weight=1 / Ntot,
        model_type="variational",
        activation_fn=simparams.nonlin,
        posterior_model="mvn",
        prior_model="isotropic",
    )
    comb = abstractmodel.build()
    
    comb.summary()
    comb.compile(
        optimizer=OPTIM,
        loss=neg_loglik,
    )
    checkpoint_filepath_combined = '/tmp/checkpoint_combined'
    model_checkpoint_callback_combined = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_combined,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
    )
    starttime = time()
    comb_history = comb.fit(
        Xtot, Ytot,
        epochs=simparams.max_iter,
        verbose=args.verbose,
        callbacks=[early_stop, model_checkpoint_callback_combined],
    )
    print(f"Elapsed time {time() - starttime:.2f}s")

    # Load best weights, make predictions. 
    # comb.load_weights(checkpoint_filepath)
    comb_preds, comb_mean, comb_ptiles = compute_prob_predictions(comb, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

    # Add predictions to plot.
    ax.plot(Xtest, comb_mean, "darkmagenta", ls="solid", lw=2.0, label="Model with OOD Data")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), comb_ptiles[0, :], comb_ptiles[1, :], color="orchid", alpha=0.5
    )

    #############################
    # MOPED Initialization
    #############################
    if args.do_moped:
        print("Training Multilayer Perceptron for MOPED")
        mlp_model = mlp(D_X=1, D_H=D_H, activation_fn=simparams.nonlin)
        mlp_model.summary()
        mlp_model.compile(
            optimizer=OPTIM,
            loss='mse',
        )
        checkpoint_filepath_moped_init = '/tmp/checkpoint_moped_init'
        model_checkpoint_callback_moped_init = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_moped_init,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True,
        )
        mlp_history = mlp_model.fit(
            Xtrain, Ytrain,
            epochs=simparams.max_iter,
            verbose=args.verbose,
            callbacks=[model_checkpoint_callback_moped_init],
        )

        # Restore best model.
        mlp_model.load_weights(checkpoint_filepath_moped_init)

        print("Iniitializing MOPED, training model.")
        moped = moped_bnn(mlp_model, kl_weight=1 / Ntot, delta=0.1, D_X=D_X, D_H=D_H, activation_fn=simparams.nonlin)
        moped.summary()
        moped.compile(
            optimizer=OPTIM,
            loss=neg_loglik,
        )
        checkpoint_filepath_moped = '/tmp/checkpoint_moped'
        model_checkpoint_callback_moped = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_moped,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True,
        )
        starttime = time()
        moped_history = moped.fit(
            Xtot, Ytot,
            epochs=simparams.max_iter,
            verbose=args.verbose,
            callbacks=[early_stop, model_checkpoint_callback_moped],
        )
        print(f"Elapsed time {time() - starttime:.2f}s")

        # Load best weights, make predictions. 
        moped.load_weights(checkpoint_filepath_moped)
        moped_preds, moped_mean, moped_ptiles = compute_prob_predictions(moped, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

        # Add predictions to plot.
        ax.plot(Xtest, moped_mean, "darkmagenta", ls="solid", lw=2.0, label="MOPED")
        # plot 90% confidence level of predictions
        ax.fill_between(
            Xtest.squeeze(), moped_ptiles[0, :], moped_ptiles[1, :], color="orchid", alpha=0.5
        )

    #############################
    # Empirical Bayes
    #############################
    if args.do_eb:
        print("Training model via Empirical Bayes.")
        abstractmodel = BNNHyperModel(
            dx=D_X,
            dh=simparams.hidden_dim,
            dy=D_Y,
            nh=simparams.num_layers,
            kl_weight=1 / Ntot,
            model_type="variational",
            activation_fn=simparams.nonlin,
            posterior_model="trainable",
            prior_model="isotropic",
        )
        eb_model = abstractmodel.build()
        eb_model.summary()
        eb_model.compile(
            optimizer=OPTIM,
            loss=neg_loglik,
        )
        checkpoint_filepath_emp_bayes = '/tmp/checkpoint_emp_bayes'
        model_checkpoint_callback_emp_bayes = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_emp_bayes,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True,
        )
        starttime = time()
        eb_history = eb_model.fit(
            Xtot, Ytot,
            epochs=simparams.max_iter,
            verbose=args.verbose,
            callbacks=[early_stop, model_checkpoint_callback_emp_bayes],
        )
        print(f"Elapsed time: {time() - starttime:.2f}s")

        # Load best weights, make predictions. 
        eb_model.load_weights(checkpoint_filepath_emp_bayes)
        eb_preds, eb_mean, eb_ptiles = compute_prob_predictions(eb_model, Xtest, num_samples=simparams.num_samples, ptiles=percentiles)

        # Add predictions to plot.
        ax.plot(Xtest, eb_mean, "darkmagenta", ls="solid", lw=2.0, label="Empirical Bayes")
        # plot 90% confidence level of predictions
        ax.fill_between(
            Xtest.squeeze(), eb_ptiles[0, :], eb_ptiles[1, :], color="orchid", alpha=0.5
        )

    #############################
    # Finalize Output
    #############################
    pretty_labels(xlabel='x', ylabel='y', fontsize=18, title=f'Mean Predictions with {simparams.percentiles:d}% Confidence Interval')
    pretty_legend(fontsize=16)
    plt.ylim((-20, 20))  # TODO: make this programmatic
    plt.savefig(os.path.join(SAVEDIR, f"{simparams.filename}.pdf"))
    plt.close()

    #############################
    # Plot only the warm-start model
    #############################
    fig, ax = plt.subplots()
    plt.scatter(Xtest, Ytest, alpha=0.5, color='gray', label='True Data')
    plt.plot(Xtrain, Ytrain, 'kx', label='High-Fidelity Training Data')
    plt.plot(Xsynth, Ysynth, 'rx', label='Low-Fidelity Training Data', alpha=0.3)
    
    # Add warmup model predictions to plot.
    ax.plot(Xtest, warmup_mean, "blue", ls="solid", lw=2.0, label="Trained Prior")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), warmup_ptiles[0, :], warmup_ptiles[1, :], color="lightblue", alpha=0.5
    )

    # Add retrained model to plot
    ax.plot(Xtest, post_mean, "orangered", ls="solid", lw=2.0, label="Warm-Start Posterior")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), post_ptiles[0, :], post_ptiles[1, :], color="lightcoral", alpha=0.7
    )

    pretty_labels(xlabel='x', ylabel='y', fontsize=18)
    pretty_legend(fontsize=16)
    plt.ylim((-20, 20))  # TODO: make this programmatic
    plt.savefig(os.path.join(SAVEDIR, "warmstartonly.pdf"))
    plt.close()

    #############################
    # Plot only the OOD model
    #############################
    fig, ax = plt.subplots()
    plt.scatter(Xtest, Ytest, alpha=0.5, color='gray', label='True Data')
    plt.plot(Xtrain, Ytrain, 'kx', label='High-Fidelity Training Data')
    plt.plot(Xood, Yood, 'rx', label='Low-Fidelity Training Data')
    
    # Add combined model predictions to plot.
    ax.plot(Xtest, comb_mean, "darkmagenta", ls="solid", lw=2.0, label="Model with OOD Data")
    # plot 90% confidence level of predictions
    ax.fill_between(
        Xtest.squeeze(), comb_ptiles[0, :], comb_ptiles[1, :], color="orchid", alpha=0.7
    )

    pretty_labels(xlabel='x', ylabel='y', fontsize=18)
    pretty_legend(fontsize=16)
    plt.ylim((-20, 20))  # TODO: make this programmatic
    plt.savefig(os.path.join(SAVEDIR, "combinedonly.pdf"))
    plt.close()

    #############################
    # Optional Plots
    #############################
    if args.save_history:
        # Warmup
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(warm_history.history['loss'])
        pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Warmup Training History")
        plt.savefig(os.path.join(SAVEDIR, "warmup_history.pdf"))
        plt.close()

        # Warm-start
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(full_history.history['loss'])
        pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Warm-start Training History")
        plt.savefig(os.path.join(SAVEDIR, "warmstart_history.pdf"))
        plt.close()

        # Combined
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(warm_history.history['loss'])
        pretty_labels(xlabel='Iteration', ylabel='ELBO', fontsize=18, title="Combined Training History")
        plt.savefig(os.path.join(SAVEDIR, "combined_history.pdf"))
        plt.close()

    if args.save_realizations:
        Nplots = simparams.num_plots
        
        # Warm-start model..
        fig, ax = plt.subplots()
        for _ in range(Nplots):
            pred = warm_start(Xtest).sample()
            plt.plot(Xtest, pred, 'k', lw=0.5)
        # plot 90% confidence level of predictions
        ax.fill_between(
            Xtest.squeeze(), post_ptiles[0, :], post_ptiles[1, :], color="gray", alpha=0.5
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Warm-Start Model Realizations')
        plt.savefig(os.path.join(SAVEDIR, "warmstart_realizations.pdf"))

        # Combined model.
        fig, ax = plt.subplots()
        for _ in range(Nplots):
            pred = comb(Xtest).sample()
            plt.plot(Xtest, pred, 'k', lw=0.5)
        # plot 90% confidence level of predictions
        ax.fill_between(
            Xtest.squeeze(), comb_ptiles[0, :], comb_ptiles[1, :], color="gray", alpha=0.5
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Combined Model Realizations')
        plt.savefig(os.path.join(SAVEDIR, "comb_realizations.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Neural Network Priors.")

    # Set up the input arguments.
    parser.add_argument("--input", help="Input deck.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--savedir", default="../Figures/extrap")
    parser.add_argument("--save-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-realizations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--do-moped", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--do-eb", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    if args.verbose:
        print(args)
    
    # ######## Run Script ########
    main(args)
