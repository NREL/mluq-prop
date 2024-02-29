#!/usr/bin/env python3
"""
This script post-processes extrapolation results and prints them to the console.
WARNING: This script is VERY sensitive to the naming convention used for the experiments.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import tensorflow as tf

from util import load_model, parse_input_deck, model2model, dns_data_loader
from prettyPlot.plotting import pretty_labels, pretty_legend

def legend_without_duplicate_labels(ax, fontsize=16, loc='upper left'):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    prop={
        #"family": "Times New Roman",
        "size": fontsize,
        "weight": "bold",
    }
    leg = ax.legend(*zip(*unique), prop=prop, loc=loc)
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")

def main(args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Make sure the figure directory exists.
    os.makedirs(args.figdir, exist_ok=True)
    
    if args.short:
        NAMES = ["Name", "FMD", "SMD", "W FMD", "W SMD", "E FMD", "E SMD", "W E FMD", "W E SMD", "Wasserstein"]
    else:
        NAMES = ["Name", "First Moment Distance", "Second Moment Distance", "Weighted First Moment Distance", "Weighted Second Moment Distance", "Extrapolatory First Moment Distance", "Extrapolatory Second Moment Distance", "Weighted Extrapolatory First Moment Distance", "Weighted Extrapolatory Second Moment Distance", "Wasserstein Distance"]
        SHORTNAMES = ["Name", "FMD", "SMD", "W FMD", "W SMD", "E FMD", "E SMD", "W E FMD", "W E SMD", "Wasserstein"]
    
    # ================================
    # Whether or not to use weighted metrics.
    if args.wmetric:
        IDX = [3,4,7,8,9]
        line = f"{NAMES[0]}"
        if args.tex:
            for i in IDX:
                line = line + f" & {NAMES[i]}" if i != 9 else line + f" & {NAMES[i]} \\\\"
        print(line)
    else:
        IDX = [1,2,5,6,9]
        line = f"{NAMES[0]}"
        if args.tex:
            for i in IDX:
                line = line + f" & {NAMES[i]}" if i != 9 else line + f" & {NAMES[i]} \\\\"
        print(line)
    
    df = pd.read_csv(args.logfile, sep=",", header=None)
    df = df.sort_values(0).reset_index(drop=True)  # organize things by trial name.

    # Loop through the dataframe and extract information about the experiment.
    vndata = []
    vtype = []
    voffset = []
    vmethname = []
    
    # ================================
    # Print tabular info to console.
    for i in range(len(df)):
        title = df[0][i]
        if title.split('_')[0] == 'NF':
            # shortname = "NF " + title.split('_')[3][:-2]
            # vndata.append(int(title.split('_')[3][:-2]))
            shortname = "NF " + title.split('_')[1][:-2]  # 10D naming convention
            vndata.append(int(title.split('_')[1][:-2]))  # 10D naming convention
            vtype.append("NF")
            voffset.append(0)
            vmethname.append("NF")
        else:
            # shortname = "SBO " + title.split('_')[1][:-2] + " " + title.split('_')[2][:-2]
            # vndata.append(int(title.split('_')[2][1:-2]))
            shortname = "SBO " + title.split('_')[1][:-2] + " " + title.split('_')[2][:]  # 10D naming convention
            vndata.append(int(title.split('_')[2][1:]))  # 10D naming convention
            vtype.append("SBO")
            voffset.append(int(title.split('_')[1][1:-2]))
            vmethname.append(f"SBO d={int(title.split('_')[1][1:-2])}")
    
        if args.tex:
            line = f"{shortname} & "
            for j in IDX:
                line = line + f"${df[j][i]:.3e}$ & " if j != 9 else line + f"${df[j][i]:.3e}$ \\\\"
            print(line)
        else:
            if i == 0:
                print("Not printing tabulated data.")

    # ================================
    # Add columns for dataset descriptors.
    df['ndata'] = vndata
    df['type'] = vtype
    df['offset'] = voffset
    df['label'] = vmethname
    
    # ================================
    # Line plot to summarize metric performance.
    # NOTE: this is heavily reliant on the naming convention and sorting.
    COLORS = pl.cm.Blues(np.linspace(0, 1, len(np.unique(df['offset']))))
    print("Generating line plots...")
    for II in IDX:
        fig, ax = plt.subplots()
        for i in range(len(np.unique(df['offset']))):
            tmp = df[df['offset'] == np.unique(df['offset'])[i]].reset_index()
            if i == 0:
                ax.loglog(tmp['ndata'], tmp[II], '-x', label=tmp['label'][0], color='k')
            else:
                ax.loglog(tmp['ndata'], tmp[II], '-x', label=tmp['label'][0], color=COLORS[i])
        
        # ax.set_ylabel(NAMES[II])
        # ax.set_xlabel("Amount of Synthetic Data")
        # ax.set_title(f"Effect of Extrapolation Method on {NAMES[II]}")
        legend_without_duplicate_labels(ax=ax, fontsize=16, loc='upper right')
        pretty_labels( 
                xlabel="Number of synthetic datapoints",
                 ylabel=f"{NAMES[II] if args.short else SHORTNAMES[II]}",
                #  title=f"Convergence of {NAMES[II]}",
                 fontsize=18,
                 yminor=True)
        plt.savefig(os.path.join(args.figdir, f'line_{II}.pdf'))
        plt.close()

    # ================================
    # Generate bar chart for comparison.
    # NOTE: this is heavily reliant on the naming convention and sorting.
    print("Generating bar plots...")
    NUM_TRIALS = 4  # number of trials per method (NF, SBO at given offset, etc.)
    width = 1 / (NUM_TRIALS + 1)  # the width of the bars
    for II in IDX:
        fig, ax = plt.subplots(layout='constrained')    
        METHODS = []
        COLORS = ["#6B5AD7", "#FF8C55", "#47D4A2", "#FFDC55"]
        ctr = 0  # counter for the x-axis.
        multiplier = 0
        for i in range(len(df)):
            if (i % NUM_TRIALS == 0) and (i != 0):
                multiplier = 0
                ctr += 1  # increment counter
            
            title = df[0][i]
            if title.split('_')[0] == 'NF':
                # num_data = int(title.split('_')[3][:-2])
                num_data = int(title.split('_')[1][:-2])  # 10D naming convention
                methname = "NF"
            else:
                sbo_offset = int(title.split('_')[1][1:-2])
                # num_data = int(title.split('_')[2][1:-2])
                num_data = int(title.split('_')[2][1:-2])  # 10D naming convention
                methname = f"SBO d={sbo_offset}"
            METHODS.append(methname) if methname not in METHODS else METHODS
            
            # We are at the beginning of a new method.
            offset = width * multiplier
            rects = ax.bar(ctr + offset, df[II][i], width, color=COLORS[i % 4], label=f"N = {num_data:.1e}")
            # ax.bar_label(rects, padding=3)
            multiplier += 1
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(NAMES[II])
        ax.set_title(f"Effect of Extrapolation Method on {NAMES[II]}")
        ax.set_xticks(np.arange(len(METHODS)) + width, METHODS)
        legend_without_duplicate_labels(ax=ax, fontsize=16)
        #pretty_legend(fontsize=16, unique=True, ax=ax)
        plt.savefig(os.path.join(args.figdir, f"bar_{II}.pdf"))
        plt.close()

    # ================================
    # Comparison to reference model.
    print("Loading in reference model...")
    
    # Load in the reference model.
    simparams = parse_input_deck(args.refinput)
    simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.
    
    # Load in data necessary for comparisons.
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
    
    # Loop through the experiments.
    vwass = []
    vrelmean = []
    vale = []
    vepi = []
    simparams = parse_input_deck(args.input)
    simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.
    for i, trial in enumerate(df[0]):
        print(f"Loading in experiment {trial}...")
        model = load_model(
            fpath=os.path.join(args.ckptdir, trial, "best/", "best"),
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
        
        wass, relmean, relale, relepi = model2model(model, refmodel, Xtest, nepi=args.npredict)
        vwass.append(wass)
        vrelmean.append(relmean)
        vale.append(relale)
        vepi.append(relepi)
        
    df['wass'] = vwass
    df['relmean'] = vrelmean
    df['ale'] = vale
    df['epi'] = vepi
    
    COLORS = pl.cm.Blues(np.linspace(0, 1, len(np.unique(df['offset']))))
    
    # Wasserstein.
    fig = plt.figure()
    for i in range(len(np.unique(df['offset']))):
        tmp = df[df['offset'] == np.unique(df['offset'])[i]].reset_index()
        if i == 0:
            plt.plot(tmp['ndata'], tmp['wass'], '-x', label=tmp['label'][0], color='k')
        else:
            plt.plot(tmp['ndata'], tmp['wass'], '-x', label=tmp['label'][0], color=COLORS[i])
    ax=plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    # ax.set_ylabel("Wasserstein Distance")
    # ax.set_xlabel("Amount of Synthetic Data")
    # ax.set_title(f"Wasserstein Distance Comparison to Reference Model")
    pretty_labels(
                 xlabel="Number of synthetic datapoints",
                 ylabel="Relative Error",
                 title="Convergence of Wasserstein Distance",
                 fontsize=18)
    #pretty_legend(fontsize=18, unique=True, ax=ax)
    legend_without_duplicate_labels(fontsize=16, ax=ax)
    plt.savefig(os.path.join(args.figdir, f'line_wass.pdf'))
    plt.close()
    
    # Relative mean.
    fig, ax = plt.subplots()
    for i in range(len(np.unique(df['offset']))):
        tmp = df[df['offset'] == np.unique(df['offset'])[i]].reset_index()
        if i == 0:
            ax.loglog(tmp['ndata'], tmp['relmean'], '-x', label=tmp['label'][0], color='k')
        else:
            ax.loglog(tmp['ndata'], tmp['relmean'], '-x', label=tmp['label'][0], color=COLORS[i])
    
    # ax.set_ylabel("Relative Mean Error")
    # ax.set_xlabel("Amount of Synthetic Data")
    # ax.set_title(f"Relative Mean Error with Reference Model")
    #pretty_legend(fontsize=16, unique=True, ax=ax)
    pretty_labels(xlabel="Number of synthetic datapoints",
                 ylabel="Relative Error",
                #  title="Convergence of Predictive Mean",
                 fontsize=16,
                 yminor=True)
    legend_without_duplicate_labels(fontsize=16, ax=ax)
    plt.savefig(os.path.join(args.figdir, f'line_relmean.pdf'))
    plt.close()
    
    # Aleatoric uncertainty.
    fig, ax = plt.subplots()
    for i in range(len(np.unique(df['offset']))):
        tmp = df[df['offset'] == np.unique(df['offset'])[i]].reset_index()
        if i == 0:
            ax.loglog(tmp['ndata'], tmp['ale'], '-x', label=tmp['label'][0], color='k')
        else:  
            ax.loglog(tmp['ndata'], tmp['ale'], '-x', label=tmp['label'][0], color=COLORS[i])
    
    # ax.set_ylabel("Relative Aleatoric Uncertainty Error")
    # ax.set_xlabel("Amount of Synthetic Data")
    # ax.set_title(f"Relative Error of Aleatoric Uncertainty with Reference Model")
    #pretty_legend(fontsize=16, unique=True, ax=ax)
    pretty_labels(xlabel="Number of synthetic datapoints",
                 ylabel="Relative Error",
                #  title="Convergence of Aleatoric Uncertainty",
                 fontsize=16,
                 yminor=True)
    legend_without_duplicate_labels(ax=ax, fontsize=16)
    plt.savefig(os.path.join(args.figdir, f'line_ale.pdf'))
    plt.close()
    
    # Epistemic uncertainty.
    fig, ax = plt.subplots()
    for i in range(len(np.unique(df['offset']))):
        tmp = df[df['offset'] == np.unique(df['offset'])[i]].reset_index()
        if i == 0:
            ax.loglog(tmp['ndata'], tmp['epi'], '-x', label=tmp['label'][0], color='k')
        else:
            ax.loglog(tmp['ndata'], tmp['epi'], '-x', label=tmp['label'][0], color=COLORS[i])
    
    # ax.set_ylabel("Relative Epistemic Uncertainty Error")
    # ax.set_xlabel("Amount of Synthetic Data")
    # ax.set_title(f"Relative Error of Epistemic Uncertainty with Reference Model")
    #pretty_legend(fontsize=18, unique=True, ax=ax)
    pretty_labels(xlabel="Number of synthetic datapoints",
                 ylabel="Relative Error",
                 title="Convergence of Epistemic Uncertainty",
                 fontsize=18)
    legend_without_duplicate_labels(ax=ax, fontsize=16)
    plt.savefig(os.path.join(args.figdir, f'line_epi.pdf'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess hyperparameter tuning runs.")
    
    parser.add_argument("--logfile", type=str, help="Absolute path to log file containing best losses and jobids.")
    parser.add_argument("--figdir", type=str, help="Directory to save figures.")
    parser.add_argument("--ckptdir", type=str, help="Directory where checkpoints are located.")
    parser.add_argument("--refinput", type=str, help="Path to reference model checkpoint.")
    parser.add_argument("--input", type=str, help="Path to reference model checkpoint.")
    parser.add_argument("--tex", action=argparse.BooleanOptionalAction, default=False, help="Print results in LaTeX format.")
    parser.add_argument("--wmetric", action=argparse.BooleanOptionalAction, default=False, help="Whether to print the moment distances weighted by the inducing point probabilities.")
    parser.add_argument("--short", action=argparse.BooleanOptionalAction, default=False, help="Print with short title.")
    parser.add_argument("--npredict", type=int, default=1, help="Number of predictions to make.")
    
    args = parser.parse_args()
    main(args)
