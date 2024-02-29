import os
import sys

import numpy as np

sys.path.append("util")
import gpfit
from fileManagement import *


def getMinMaxVal(np_data):
    return np.amin(np_data, axis=0), np.amax(np_data, axis=0)


def rescaleData(np_data, minVal, maxVal):
    np_data_rescaled = np_data.copy()
    np_data_rescaled = (np_data_rescaled - minVal) / (
        0.125 * (maxVal - minVal)
    ) - 4
    return np_data_rescaled


def unrescaleData(np_data, minVal, maxVal):
    np_data_rescaled = np_data.copy()

    np_data_rescaled = (np_data + 4) * 0.125 * (maxVal - minVal) + minVal

    return np_data_rescaled


def get_gp(data, qty, minVal, maxVal):
    x = data["loc"]
    y = data[qty]
    x_resc = rescaleData(x, minVal, maxVal)
    mean, _, gpr, _ = gpfit.gpfit_simple(
        x_resc,
        y,
        ndim=x_resc.shape[1],
        n_restart=20,
        constant_value_bounds=(1e-4, 10),
        length_scale_bounds=(1e-10, 1e3),
    )
    mean, _, gpr, _ = gpfit.gpfit_homo(
        x_resc,
        y,
        ndim=x_resc.shape[1],
        n_restart=20,
        constant_value_bounds=(1e-4, 10),
        noise_level_bounds=(1e-10, 1e3),
    )
    return gpr


def call_gp(gp, data_in, minVal, maxVal):
    data_in_resc = rescaleData(data_in, minVal, maxVal)
    y_interp = gp.predict(data_in_resc, return_std=False)
    return y_interp


dataFolder = "data"

dataTest = np.load(os.path.join(dataFolder, "downSampledData_10000_best.npz"))[
    "data"
]
cluster_list = getClusterList(dataFolder)

data_list = [
    np.load(os.path.join(dataFolder, f"frechetDistRef_{cluster}.npz"))
    for cluster in cluster_list
]
minVal, maxVal = getMinMaxVal(dataTest)

meanGP = []
epistGP = []
aleatGP = []

mean_pred = []
epist_pred = []
aleat_pred = []

for idata, data in enumerate(data_list):
    gp_mean = get_gp(data, "mean", minVal, maxVal)
    gp_epist = get_gp(data, "epist", minVal, maxVal)
    gp_aleat = get_gp(data, "aleat", minVal, maxVal)

    print(f"Cluster {cluster_list[idata]}")
    print(f"\t mean {gp_mean.kernel_}")
    print(f"\t epist {gp_epist.kernel_}")
    print(f"\t aleat {gp_aleat.kernel_}")

    meanGP.append(gp_mean)
    epistGP.append(gp_epist)
    aleatGP.append(gp_aleat)

    mean_pred.append(call_gp(gp_mean, dataTest, minVal, maxVal))
    epist_pred.append(call_gp(gp_epist, dataTest, minVal, maxVal))
    aleat_pred.append(call_gp(gp_aleat, dataTest, minVal, maxVal))

np.savez(
    "convergence.npz",
    mean=mean_pred,
    epist=epist_pred,
    aleat=aleat_pred,
    cluster=cluster_list,
)
