#!/usr/bin/env python3
"""
This script post-processes extrapolation results and prints them to the console.
WARNING: This script is VERY sensitive to the naming convention used for the experiments.
WARNING: This script is VERY brittle. It is not meant to be used for anything other than the 10D extrapolation experiments.
"""

import os
import time
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import tensorflow as tf

from prettyPlot.plotting import pretty_labels

from mluqprop.BNN.util.dns import load_extrap_data, dns_data_loader
from mluqprop.BNN.util.input_parser import parse_input_deck
from mluqprop.BNN.util.models import BNNHyperModel, load_model
from mluqprop.BNN.util.metrics import model2model, compute_model_moments_online

def legend_without_duplicate_labels(ax, fontsize=16, loc='upper left'):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    prop={
        "family": "serif",
        "size": fontsize,
        "weight": "bold",
    }
    leg = ax.legend(*zip(*unique), prop=prop, loc=loc)
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")
    
    
def buildModel(DX, DY, simparams):
    # Create model.
    abstractmodel = BNNHyperModel(dx=DX,
                                  dh=simparams.hidden_dim,
                                  dy=DY,
                                  nh=simparams.num_layers,
                                  kl_weight=-1,  # no training.
                                  model_type=simparams.model_type,
                                  activation_fn=simparams.nonlin,
                                  posterior_model=simparams.posterior_model,
                                  prior_model=simparams.prior_model,
                                  split=simparams.split
                                )
    model = abstractmodel.build()
    
    return model


def main(args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
    
    #############################
    # Reference model.
    #############################
    simparams = parse_input_deck(args.refinput)
    simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.
    
    # Load in data and reference model for calculations.
    _, _, Xtest, Ytest = dns_data_loader(simparams.data_fpath)
    N, DX = Xtest.shape
    DY = Ytest.shape[1]
    Xtest = Xtest[::25]  # aggressive downsampling for statistics computation.
    
    refmodel = load_model(
            fpath=os.path.join(simparams.checkpath, "best/", "best"),
            D_X=DX,
            D_H=simparams.hidden_dim,
            D_Y=DY,
            N_H=simparams.num_layers,
            kl_weight=1.0,  # dummy value
            model_type=simparams.model_type,
            activation_fn=simparams.nonlin,
            posterior_model=simparams.posterior_model,
            split=simparams.split,
        )
    print(f"Reference model loaded from {simparams.checkpath}.")
    
    uips = np.load(simparams.uips_fpath)
    
    #############################
    # Parse the input deck.
    #############################
    if args.input is None:
        raise NameError("No input deck provided. Please specify with: --input <input_deck>")
    simparams = parse_input_deck(args.input)
    simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.

    # Make sure the figure directory exists.
    os.makedirs(args.figdir, exist_ok=True)
    
    # Set up dictionary to hold model information.
    MODELINFO = {}
    NDATA = []
    OFFSET = []
    
    # Loop through models and extract information.
    start = time.perf_counter()
    model_names = os.listdir(args.ckptdir)
    for mname in model_names:
        if mname.split('_')[0] == 'NF':
            # Normalizing Flow.
            ndata = mname.split('_')[1][:-2]  # 10D naming convention
            NDATA.append(ndata)
            
            # Pack information into dictionary.
            MODELINFO[mname] = {}
            MODELINFO[mname]['type'] = "NF"
            MODELINFO[mname]['shortname'] = f"NF {ndata}"
            MODELINFO[mname]['ndata'] = int(ndata)
    
            # Set extrapolation data name.
            # TODO: make programmatic
            EXTRAPDATA = os.path.join(args.extrap_datapath, "NF_genBack", "backgroundComb_highD_labeled_fixed.npz")
            
            # Load in extrapolation inducing points.
            extrap_uips = np.load(os.path.join(args.extrap_datapath, "NF_genBack", "frechetDistRef_synth_100_fixed.npz"))
        else:
            # Soft-Brownian Offset.
            offset = mname.split('_')[1][1:-2]  # 10D naming convention
            ndata = mname.split('_')[2][1:]  # 10D naming convention
            
            NDATA.append(ndata)
            OFFSET.append(offset)
            
            # Pack information into dictionary.
            MODELINFO[mname] = {}
            MODELINFO[mname]['type'] = "SBO"
            MODELINFO[mname]['shortname'] = f"SBO d{offset} n{ndata}"
            MODELINFO[mname]['ndata'] = int(ndata)
            MODELINFO[mname]['offset'] = int(offset)
            
            # Set extrapolation data name.
            # TODO: make programmatic
            EXTRAPDATA = os.path.join(args.extrap_datapath, "SBO_genBack_uips", f"{mname}.npz")
            
            # Load in extrapolation inducing points.
            uipsname = mname
            # NOTE: THIS IS A HACK TO GET THE CORRECT FILENAME.
            tmp = uipsname.split('_')
            tmp[-2] = "100"
            uipsname = '_'.join(tmp)
            extrap_uips = np.load(os.path.join(args.extrap_datapath, "SBO_genBack_uips", f"frechetDistRef_{uipsname}.npz"))
        
        # Load extrapolation data.
        Xtrain, Ytrain, Xval, Yval, Xextrap, Yextrap = load_extrap_data(simparams.data_fpath, EXTRAPDATA, MODELINFO[mname]['ndata'])
        
        # Load weights into model.
        _, DX = Xtrain.shape
        DY = Ytrain.shape[1]
        model = buildModel(DX, DY, simparams)
        model.load_weights(os.path.join(args.ckptdir, mname, "best/", "best"))
        
        # Comparison to reference model.
        start_comp = time.perf_counter()
        _, relmean, relale, relepi = model2model(model, refmodel, Xtest, nepi=args.npredict)
        MODELINFO[mname]['relmean'] = relmean
        MODELINFO[mname]['relale'] = relale
        MODELINFO[mname]['relepi'] = relepi
        print(f"Time for model comparison:\t{time.perf_counter() - start_comp:.2f} seconds.")
        
        # Extrapolatory moment distances.
        # mdextrap, sdextrap = vec_moment_disance(model, uips=extrap_uips, nalea=simparams.nalea, nepi=simparams.nepi)

        start_moment = time.perf_counter()
        pmean, _, pale = compute_model_moments_online(model, Xextrap, nepi=args.npredict)
        mdextrap = np.linalg.norm(pmean, ord=2) / np.sqrt(len(Yextrap))  # since the extrapolation data come from N(0, \sigma_glob^2)
        sdextrap = np.linalg.norm(pale - np.std(Yextrap), ord=2) / np.sqrt(len(Yextrap))
        print(f"Time to compute model moments:\t{time.perf_counter() - start_moment:.2f} seconds.")
        
        MODELINFO[mname]['mdextrap'] = mdextrap
        MODELINFO[mname]['sdextrap'] = sdextrap
    
    print(f"Computing metrics took:\t{time.perf_counter() - start:.2f} seconds.")
    
    # Print info to console.
    if args.verbose:
        pprint.pprint(MODELINFO)
    
    print("Making figures...")

    #############################
    # Plot the comparison with the reference model mean.
    #############################
    # First loop through the models and extract the convergence information.
    # TODO: do this in a more intelligent manner
    uDATA = np.unique(NDATA)
    xDATA = np.unique(NDATA).astype('int')
    uOFFSET = np.unique(OFFSET)
    
    NFDATA = np.zeros((len(uDATA), 1))
    SBODATA = np.zeros((len(uDATA), len(uOFFSET)))

    for mname in model_names:
        i = np.where(uDATA == str(MODELINFO[mname]['ndata']))[0][0]
        if MODELINFO[mname]['type'] == "NF":
            NFDATA[i] = MODELINFO[mname]['relmean']
        if MODELINFO[mname]['type'] == "SBO":
            j = np.where(uOFFSET == str(MODELINFO[mname]['offset']))[0][0]
            SBODATA[i, j] = MODELINFO[mname]['relmean']

    COLORS = pl.cm.Blues(np.linspace(0.2, 0.9, len(uOFFSET)))
    fig, ax = plt.subplots()
    ax.semilogx(xDATA, NFDATA, '-kx', label="NF")
    for i in range(len(uOFFSET)):
        ax.semilogx(xDATA, SBODATA[:, i], '-x', label=f"SBO d={uOFFSET[i]}", color=COLORS[i])
    legend_without_duplicate_labels(ax=ax, fontsize=16, loc='upper left')
    pretty_labels( 
                xlabel="Number of synthetic datapoints",
                ylabel="Relative Error",
                xminor=True,
                yminor=True,
                fontsize=18)
    plt.savefig(os.path.join(args.figdir, f"line_relmean.pdf"))
    plt.close()
    
    #############################
    # Plot the comparison with the reference model (aleatoric) variance.
    #############################
    for mname in model_names:
        i = np.where(uDATA == str(MODELINFO[mname]['ndata']))[0][0]
        if MODELINFO[mname]['type'] == "NF":
            NFDATA[i] = MODELINFO[mname]['relale']
        if MODELINFO[mname]['type'] == "SBO":
            j = np.where(uOFFSET == str(MODELINFO[mname]['offset']))[0][0]
            SBODATA[i, j] = MODELINFO[mname]['relale']

    fig, ax = plt.subplots()
    ax.semilogx(xDATA, NFDATA, '-kx', label="NF")
    for i in range(len(uOFFSET)):
        ax.semilogx(xDATA, SBODATA[:, i], '-x', label=f"SBO d={uOFFSET[i]}", color=COLORS[i])
    legend_without_duplicate_labels(ax=ax, fontsize=16, loc='upper left')
    pretty_labels( 
                xlabel="Number of synthetic datapoints",
                ylabel="Relative Error",
                xminor=True,
                yminor=True,
                fontsize=18)
    plt.savefig(os.path.join(args.figdir, f"line_relale.pdf"))
    plt.close()
    
    #############################
    # Plot the comparison with the reference model (epistemic) variance.
    #############################
    for mname in model_names:
        i = np.where(uDATA == str(MODELINFO[mname]['ndata']))[0][0]
        if MODELINFO[mname]['type'] == "NF":
            NFDATA[i] = MODELINFO[mname]['relepi']
        if MODELINFO[mname]['type'] == "SBO":
            j = np.where(uOFFSET == str(MODELINFO[mname]['offset']))[0][0]
            SBODATA[i, j] = MODELINFO[mname]['relepi']

    fig, ax = plt.subplots()
    ax.semilogx(xDATA, NFDATA, '-kx', label="NF")
    for i in range(len(uOFFSET)):
        ax.semilogx(xDATA, SBODATA[:, i], '-x', label=f"SBO d={uOFFSET[i]}", color=COLORS[i])
    legend_without_duplicate_labels(ax=ax, fontsize=16, loc='upper left')
    pretty_labels( 
                xlabel="Number of synthetic datapoints",
                ylabel="Relative Error",
                xminor=True,
                yminor=True,
                fontsize=18)
    plt.savefig(os.path.join(args.figdir, f"line_relepi.pdf"))
    plt.close()
    
    #############################
    # Plot convergence of moments extrapolation region.
    #############################
    for mname in model_names:
        i = np.where(uDATA == str(MODELINFO[mname]['ndata']))[0][0]
        if MODELINFO[mname]['type'] == "NF":
            NFDATA[i] = MODELINFO[mname]['mdextrap']
        if MODELINFO[mname]['type'] == "SBO":
            j = np.where(uOFFSET == str(MODELINFO[mname]['offset']))[0][0]
            SBODATA[i, j] = MODELINFO[mname]['mdextrap']

    fig, ax = plt.subplots()
    ax.loglog(xDATA, NFDATA, '-kx', label="NF")
    for i in range(len(uOFFSET)):
        ax.loglog(xDATA, SBODATA[:, i], '-x', label=f"SBO d={uOFFSET[i]}", color=COLORS[i])
    legend_without_duplicate_labels(ax=ax, fontsize=16, loc='lower left')
    pretty_labels( 
                xlabel="Number of synthetic datapoints",
                ylabel="Normalized Error",
                xminor=True,
                yminor=True,
                fontsize=18)
    plt.savefig(os.path.join(args.figdir, f"line_mean_error_extrap.pdf"))
    plt.close()
    
    for mname in model_names:
        i = np.where(uDATA == str(MODELINFO[mname]['ndata']))[0][0]
        if MODELINFO[mname]['type'] == "NF":
            NFDATA[i] = MODELINFO[mname]['sdextrap']
        if MODELINFO[mname]['type'] == "SBO":
            j = np.where(uOFFSET == str(MODELINFO[mname]['offset']))[0][0]
            SBODATA[i, j] = MODELINFO[mname]['sdextrap']

    fig, ax = plt.subplots()
    ax.loglog(xDATA, NFDATA, '-kx', label="NF")
    for i in range(len(uOFFSET)):
        ax.loglog(xDATA, SBODATA[:, i], '-x', label=f"SBO d={uOFFSET[i]}", color=COLORS[i])
    legend_without_duplicate_labels(ax=ax, fontsize=16, loc='lower left')
    pretty_labels( 
                xlabel="Number of synthetic datapoints",
                ylabel="Normalized Error",
                xminor=True,
                yminor=True,
                fontsize=18)
    plt.savefig(os.path.join(args.figdir, f"line_std_error_extrap.pdf"))
    
    print("Finished making figures. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess hyperparameter tuning runs.")
    
    # Path to reference model checkpoint (shared extrap input deck).
    parser.add_argument("--input", type=str, help="Path to reference model checkpoint.")    
    
    # Extrapolation data paths.
    parser.add_argument("--extrap_datapath", default="/projects/safvto/mhassana/" ,type=str, help="Path to extrapolation data.")
    parser.add_argument("--nf_extrap_fname", default="backgroundComb_highD_labeled_fixed.npz", type=str, help="Path to NF extrapolation data.")
    
    # Reference model, output directory, models.
    parser.add_argument("--figdir", type=str, help="Directory to save figures.")
    parser.add_argument("--ckptdir", type=str, help="Directory where checkpoints are located.")
    parser.add_argument("--refinput", type=str, help="Path to reference model checkpoint.")

    # Metric computation options.
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Increase verbosity of output.")
    parser.add_argument("--npredict", type=int, default=1, help="Number of predictions to make.")
    
    args = parser.parse_args()
    main(args)
