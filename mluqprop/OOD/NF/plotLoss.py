import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from prettyPlot.plotting import plt, pretty_labels
from prettyPlot.parser import parse_input_file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import argparse
parser = argparse.ArgumentParser(description="plotter")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    metavar="",
    required=False,
    help="Input file",
    default="input",
)
args, unknown = parser.parse_known_args()


inpt = parse_input_file(args.input)

if inpt["pdf_method"].lower() == "normalizingflow":
    nIter = int(inpt["num_pdf_iter"])
    
    # Folder where figures are saved
    figureFolder = "Figures"
    os.makedirs(figureFolder, exist_ok=True)
    
    
    Loss = np.genfromtxt(
        f"TrainingLog/log_iter0.csv", delimiter=";", skip_header=1
    )
    fig=plt.figure()
    plt.plot(Loss[:, 0], Loss[:, 1], color="k", linewidth=3)
    pretty_labels("Step", "Loss", 14, title=f"iteration 0")
    plt.savefig(figureFolder + "/loss.png")
    plt.close()
