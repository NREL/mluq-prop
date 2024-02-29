#!/usr/bin/env python

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
    model = load_model(
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
    model.summary()


    PERCENTILES = [5, 95]
    print("Making epistemic uncertainty predictions...")
    print(f"Number of data points: {Xtest.shape[0]}")
    print(f"Number of predictions: {args.npredict}")
    start = time.time()
    
    #  we need the array of predictions to compute the epistemic uncertainty in physical space
    predicted, aleatory = compute_raw_epi_predictions(
            model, Xtest, num_samples=args.npredict
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
    preds, preds_mean, preds_ptile = compute_prob_predictions(model, Xtest, num_samples=args.npredict, ptiles=PERCENTILES)
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


    # Plot the training histories.
    print("Processing model training history...")
    try:
        metric_history = np.genfromtxt(os.path.join(simparams.checkpath, "metrics.log"), delimiter=",")
        history = {}
        history['epochs'] = metric_history[:, 0]
        history['wasserstein'] = metric_history[:, 1]
        history['medsnr'] = metric_history[:, 2]
        history['meandist'] = metric_history[:, 3]
        history['stddist'] = metric_history[:, 4]
            
        
        # Wasserstein Distance.
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(history['epochs'], history['wasserstein'])
        pretty_labels(xlabel='Epoch', ylabel='Wasserstein Distance', fontsize=18)
        plt.savefig(os.path.join(simparams.savedir, f"{simparams.filename}_wassersteinhistory.pdf"))
        plt.close()
        
        # Median SNR.
        fig, ax = plt.subplots()
        plt.plot(history['epochs'], history['medsnr'])
        pretty_labels(xlabel='Epoch', ylabel='Median SNR (dB)', fontsize=18)
        plt.savefig(os.path.join(simparams.savedir, f"{simparams.filename}_snrhistory.pdf"))
        plt.close()
        
        # First moment distance.
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(history['epochs'], history['meandist'])
        pretty_labels(xlabel='Epoch', ylabel="Relative Error of First Moment", fontsize=18)
        plt.savefig(os.path.join(simparams.savedir, f"{simparams.filename}_meanhistory.pdf"))
        plt.close()
        
        # Second moment distance.
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        plt.plot(history['epochs'], history['stddist'])
        pretty_labels(xlabel='Epoch', ylabel="Relative Error of Second Moment", fontsize=18)
        plt.savefig(os.path.join(simparams.savedir, f"{simparams.filename}_stdhistory.pdf"))
        plt.close()
    except:
        print("WARNING: No training history found. Skipping plotting of training history.")

    # Try to load in and plot the model training history.
    try:
        with open(os.path.join(simparams.simparams.savedir, f"{simparams.filename}_history.pkl"), "rb") as f:
            model_history = pickle.load(f)
        plt.figure()
        plt.plot(model_history['loss'])
        pretty_labels(xlabel='Epoch', ylabel='Loss', fontsize=18, title="Training History")
        plt.savefig(os.path.join(simparams.simparams.savedir, f"{simparams.filename}_loss.png"))
        plt.close()
    except:
        print("WARNING: Model history could not be loaded. Skipping plotting of model history.")


    # TODO: SNR utility for split models.
    if simparams.split == False:
        # Plot the signal to noise ratio CDF and PDF.
        if simparams.model_type == "flipout":
            snr = compute_snr_flipout(model)
        else:
            snr = compute_snr(model)

        print("Evaluating Signal to Noise Ratio...")
        # Histogram and KDE.
        plt.figure()
        sns.displot(snr, kde=True, bins=len(snr)//10, stat='probability')
        pretty_labels(xlabel="Signal-to-Noise Ratio (dB)", ylabel="Probability", fontsize=18)
        plt.savefig(os.path.join(simparams.savedir, f"snr_pdf_{simparams.filename}.pdf"))
        plt.close()
        
        # Empirical CDF.
        plt.figure()
        sns.ecdfplot(snr, color='k')
        pretty_labels(xlabel="Signal-to-Noise Ratio (dB)", ylabel="Cumulative Probability", fontsize=18)
        plt.savefig(os.path.join(simparams.savedir, f"snr_cdf_{simparams.filename}.pdf"))
        plt.close()

    
    print("Making hex-scatter plot for the predictive mean.")
    PlotLogHexScatter(simparams, Ytest, np.array(epipreds_mean))

    print("Making scatter plot shaded by uncertainties.")
    PlotScatterMeanWithUncertainty(simparams, Ytest, epipreds_mean, epistd, "Epistemic")
    PlotScatterMeanWithUncertainty(simparams, Ytest, epipreds_mean, alestd, "Aleatoric")

    # TODO maybe update this?
    print("Predicting at Inducing Points...")
    uips = np.load(simparams.uips_fpath)
    _, pmean, pepi, pale = vec_model_moments(model, uips["loc"], nalea=simparams.nalea, nepi=args.npredict)
    pmean = pmean[:, np.newaxis]
    pepi = pepi[:, np.newaxis]
    uips_mean = uips["mean"][:, np.newaxis]
    uips_aleat = uips["aleat"][:, np.newaxis]
    
    if args.rescale:
        pmean = inv_scale_otpt_lrm(pmean, uips["loc"])
        pepi = inv_scale_uncertainty(pepi)
        pale = inv_scale_uncertainty(pale)
        uips_aleat = inv_scale_uncertainty(uips_aleat)
        uips_mean = inv_scale_otpt_lrm(uips_mean, uips["loc"])
        if args.errorbars:
            uipseb = inv_scale_otpt_lrm(uips["mean_error"], uips["loc"])
        else:
            uipseb = None
    
    print("Making scatter plot shaded by uncertainties for inducing points.")
    PlotInducingScatter(simparams, uips_mean, pmean, pepi, "inducing_epi", r"Epistemic Uncertainty $[-]$", errorbars=args.errorbars, uipseb=uipseb, xlab=r"cluster mean $[-]$", ylab=r"cluster mean $[-]$")
    PlotInducingScatter(simparams, uips_mean, pmean, np.log(uips["loc_prob"]), "inducing_prob", "Log Probability", errorbars=args.errorbars, uipseb=uipseb, xlab=r"cluster mean $[-]$", ylab=r"cluster mean $[-]$")
    
    PlotInducingScatter(simparams, uips_aleat, pale, pepi, "inducing_uncertainty", r"Epistemic Uncertainty $[-]$", errorbars=args.errorbars, uipseb=uips["aleat_err"], xlab=r"cluster stddev $[-]$", ylab=r"cluster stddev $[-]$")
    PlotInducingScatter(simparams, uips_aleat, pale, np.log(uips["loc_prob"]), "inducing_uncertainty_prob", "Log Probability", errorbars=args.errorbars, uipseb=uips["aleat_err"], xlab=r"cluster stddev $[-]$", ylab=r"cluster stddev $[-]$")
    

    wrlmu = weighted_norm(uips_mean-pmean, uips["loc_prob"]) / weighted_norm(uips_mean, uips["loc_prob"])
    wrlsig = weighted_norm(uips_aleat-pale, uips["loc_prob"]) / weighted_norm(uips_aleat, uips["loc_prob"])
    print(f"Weighted Relative L2 Norm of First Moment: {wrlmu:.3e}.")
    print(f"Weighted Relative L2 Norm of Second Moment: {wrlsig:.3e}.")

    
    # Plot the conditional means with credible intervals.
    if args.condmean:
        print("Plotting Conditional Means with Credible Intervals...")
        # Rescale the data to physical space.
        xtestphysical = inv_scale_inpt(Xtest)
        if args.datatype == "lrm":
            
            # convert the predictions to physical space, compute the epistemic uncertainty.
            for i in range(predicted.shape[0]):
                predicted[i, :] = inv_scale_otpt_lrm(predicted[i, :][:, np.newaxis], Xtest)[:, 0]
            epistdphysical = np.std(predicted, axis=0)[:, np.newaxis]
            # Epistemic uncertainty estimate.
            # epistdphysical = inv_scale_otpt_lrm(epistd_orig, Xtest)
            
            # Linear relaxation model. Target = Y - LRM(X)
            ytestphysical = inv_scale_otpt_lrm(Ytest_orig, Xtest)
            # Mean prediction.
            ypredphysical = inv_scale_otpt_lrm(epipreds_mean_orig, Xtest)
            # Full predictive envelope.
            ypredptilephysicallo = inv_scale_otpt_lrm(preds_ptile_orig[0,:].T[:, np.newaxis], Xtest)
            ypredptilephysicalhi = inv_scale_otpt_lrm(preds_ptile_orig[1,:].T[:, np.newaxis], Xtest)
            # Epistemic predictive envelope.
            yepiepredptilphysicallo = inv_scale_otpt_lrm(epi_ptile_orig[0,:].T[:, np.newaxis], Xtest)
            yepiepredptilphysicalhi = inv_scale_otpt_lrm(epi_ptile_orig[1,:].T[:, np.newaxis], Xtest)
            # Aleatory uncertainty estimate.
            alestdphysical = inv_scale_uncertainty(alestd_orig)
            # LRM predictions.
            # ylrmphysical = inv_scale_otpt(computeLRM_fromScaled(Xtest))
        elif args.datatype == "normalized":
            # Normalized DNS data. Target = Y - mu / sigma
            ytestphysical = inv_scale_otpt_stdn(Ytest_orig)
            ypredphysical = inv_scale_otpt_stdn(preds_mean_orig, Xtest)
            ypredptilephysicallo = inv_scale_otpt_stdn(preds_ptile_orig[0,:].T[:, np.newaxis], Xtest)
            ypredptilephysicalhi = inv_scale_otpt_stdn(preds_ptile_orig[1,:].T[:, np.newaxis], Xtest)
        elif args.datatype == "dns":
            # Scaled dataset provided by Shashank. Target = Y
            ytestphysical = inv_scale_otpt(Ytest_orig)
            ypredphysical = inv_scale_otpt(preds_mean_orig, Xtest)
            ypredptilephysicallo = inv_scale_otpt(preds_ptile_orig[0,:].T[:, np.newaxis], Xtest)
            ypredptilephysicalhi = inv_scale_otpt(preds_ptile_orig[1,:].T[:, np.newaxis], Xtest)
        else:
            raise ValueError("Invalid datatype. Please specify the datatype to be rescaled to physical space.")

        # Compute the conditional means and plot.
        NBIN = 250
                
        if args.all_condmeans:
            indices = range(D_X)
            names = [r"$\widetilde{C} [-]$", r"$\widetilde{C^{''2}} \frac{1}{\rm{maxC} (1 - \rm{maxC})} [-]$", r"$\widetilde{\chi}/\chi_{\rm{lam}} [-]$", r"$\alpha / \chi_{\rm{lam}}$", r"$\beta / \chi_{\rm{lam}} $", r"$\gamma / \chi_{\rm{lam}}$", "Dot Grad C with Grad Eigenvector I", "Dot Grad C with Grad Eigenvector II", "Dot Grad C with Grad Eigenvector III", r"$\widetilde{D}_{C} / D_{\rm un}[-]$", "Dot Grad T Grad C", "Filtered Reaction Source"]
            shortnames = ["fc", "fcvar", "chicf", "alpha", "beta", "gamma", "gradalphagradc", "gradbetagradc", "gradgammagradc", "fd", "gradtgradc", "fomegac"]
        else:
            indices = [0, 1, 2, 9]
            names = [r"$\widetilde{C} [-]$", r"$\widetilde{C^{''2}} \frac{1}{\rm{maxC} (1 - \rm{maxC})} [-]$", r"$\widetilde{\chi} / \chi_{\rm lam} [-]$", r"$\widetilde{D}_{C} / D_{\rm unb} [-]$"]
            shortnames = ["fc", "fcvar", "chicf", "fd"]
        
        j = 0  # for grabbing the names.
        for i in indices:
            print(f"Computing conditional means for: {names[j]}...")
            start = time.time()
            # Compute the conditional means.
            xbindata, ybinvaldata = conditionalAverage(xtestphysical[:, i][:, np.newaxis], ytestphysical, NBIN)
            xbinpred, ybinpred = conditionalAverage(xtestphysical[:, i][:, np.newaxis], ypredphysical, NBIN)
            _, ybinpredptilelo = conditionalAverage(xtestphysical[:, i][:, np.newaxis], ypredptilephysicallo, NBIN)
            _, ybinpredptilehi = conditionalAverage(xtestphysical[:, i][:, np.newaxis], ypredptilephysicalhi, NBIN)
            _, ybinepipredptilelo = conditionalAverage(xtestphysical[:,i][:,np.newaxis], yepiepredptilphysicallo, NBIN)
            _, ybinepipredptilehi = conditionalAverage(xtestphysical[:,i][:,np.newaxis], yepiepredptilphysicalhi, NBIN)
            _, ybinepistd = conditionalAverage(xtestphysical[:,i][:,np.newaxis], epistdphysical, NBIN)
            _, ybinalestd = conditionalAverage(xtestphysical[:,i][:,np.newaxis], alestdphysical, NBIN)
            # _, ybinlrm = conditionalAverage(xtestphysical[:,i][:,np.newaxis], ylrmphysical, NBIN)
            print(f"Elapsed conditional averaging time: {time.time()-start:.2f}")
        
            # Plot the confidence intervals.
            print(f"Plotting confidence intervals for: {names[j]}")
            PlotConfidenceInterval(xtestphysical[:,i][:,np.newaxis], ytestphysical, xbindata, ybinvaldata, ybinpred, ybinepistd, ybinalestd, names[j], shortnames[j], simparams)
            
            print(f"Plotting the conditional data hexbin for: {names[j]}")
            PlotConditionalDataLogHex(xtestphysical[:,i][:,np.newaxis], ytestphysical, xbindata, ybinvaldata, names[j], shortnames[j], simparams)
            
            # Plot the credible intervals.
            print(f"Plotting credible intervals for: {names[j]}")
            PlotCredibleInterval(xbindata, ybinvaldata, xbinpred, ybinpred, ybinpredptilelo, ybinpredptilehi, names[j], shortnames[j], simparams)

            if i == 0:
                print("Plotting credible intervals with inset")
                PlotCredibleIntervalInset(xbindata, ybinvaldata, xbinpred, ybinpred, ybinpredptilelo, ybinpredptilehi, ybinepipredptilelo, ybinepipredptilehi,names[j], shortnames[j], simparams)
            
            print(f"Plotting epistemic confidence intervals for: {names[j]}")
            PlotConditionalStdDev(xbindata, ybinepistd, names[j], shortnames[j], simparams, savename="epi")
            
            print(f"Plotting aleatoric conditional mean for: {names[j]}")
            PlotConditionalStdDev(xbindata, ybinalestd, names[j], shortnames[j], simparams, savename="ale")
           
            print(f"Plotting epistemic and aleatoric confidence intervals for: {names[j]}")
            PlotConditionalStdDev([xbindata, xbindata], [ybinepistd*100, ybinalestd],  names[j], shortnames[j], simparams, savename="epi_ale", labels=[r'Epistemic $\times$ 100', 'Aleatoric'], colors=['r', 'b'])
  
            j = j + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BNN model performance on DNS data.")

    parser.add_argument("--input", help="Input deck.")
    parser.add_argument("--npredict", type=int, default=100)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True, help="Rescale data to physical space.")
    parser.add_argument("--errorbars", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to include error bars on the inducing points plots.")
    parser.add_argument("--condmean", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to compute the conditional means.")
    parser.add_argument("--datatype", default="lrm", choices=["lrm", "dns", "normalized"], help="The type of data being used in evaluation, for scaling purposes.")
    parser.add_argument("--all_condmeans", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to compute the conditional means for all variables.")

    args = parser.parse_args()

    # ######## Run Script ########
    main(args)
