import sys
import os
import numpy as np

import matplotlib.pyplot as plt

from prettyPlot.parser import parse_input_file
from prettyPlot.plotting import pretty_labels, pretty_legend

def plotScatterProjection(orig, background, fieldNames, lims):
    nDim = orig.shape[1]
    if nDim > 2:
        fig, axs = plt.subplots(
            nDim - 1, nDim - 1, figsize=(12, 12), sharex="col", sharey="row"
        )
        for idim in range(nDim - 1):
            for jdim in range(idim + 1, nDim):
                # plot contours of support of all data
                #a = axs[jdim - 1, idim].scatter(
                #    orig[:, idim], orig[:, jdim], color="k", s=0.2
                #)
                a = axs[jdim - 1, idim].scatter(
                    background[:, idim], background[:, jdim], color="r", s=0.2
                )

        for idim in range(nDim - 1):
            axs[nDim - 2, idim].set_xlabel(fieldNames[idim])
            axs[nDim - 2, idim].set_xlim(lims[idim])
            for tick in axs[nDim - 2, idim].get_xticklabels():
                tick.set_rotation(33)
            axs[idim, 0].set_ylabel(fieldNames[idim + 1])
            axs[idim, 0].set_ylim(lims[idim + 1])

        for idim in range(nDim - 2):
            for jdim in range(idim + 1, nDim - 1):
                axs[idim, jdim].axis("off")
    if nDim == 2:
        fig = plt.figure()
        plt.scatter(orig[:, 0], orig[:, 1], color="k", s=0.2)
        plt.scatter(background[:, 0], background[:, 1], color="r", s=0.5)
        ax = plt.gca()
        pretty_labels(fieldNames[0], fieldNames[1], 14, ax=ax)
        ax.set_xlim(lims[0])
        for tick in ax.get_xticklabels():
            tick.set_rotation(33)
        ax.set_ylim(lims[1])


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
# Data file name
fullDataFile = inpt["dataFile"]
# Scaler file name
scalerFile = inpt["scalerFile"]
print("LOAD DATA ... ", end="")
# BackgroundData
output = np.load(f"{inpt['prefixBackgroundData']}.npz")
origData = output["origData"]
backgroundData = output["backgroundData"]
sys.stdout.flush()
print("DONE!")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mins = np.amin(backgroundData, axis=0)
maxs = np.amax(backgroundData, axis=0)
lims = [
    (xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    for xmin, xmax in zip(mins, maxs)
]
fieldNames = ["feature" + str(i) for i in range(origData.shape[1])]

# Folder where figures are saved
figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)

plotScatterProjection(origData, backgroundData, fieldNames, lims)
plt.savefig(os.path.join(figureFolder, inpt["prefixBackgroundData"] + '.png'))
#plt.close()
plt.show()
