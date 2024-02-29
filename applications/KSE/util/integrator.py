import numpy as np
import tensorflow as tf

import sys
import os

from mluqprop.BNN.util.models import compute_predictions

def l96StepRK2(u, Sim):
    utmp = u.copy()
    source = (
        u[Sim["im"], :] * (u[Sim["ip"], :] - u[Sim["im2"], :]) + Sim["R"] - u
    )
    k1 = utmp + Sim["Timestep"] * 0.5 * source
    source = (
        k1[Sim["im"], :] * (k1[Sim["ip"], :] - k1[Sim["im2"], :])
        + Sim["R"]
        - k1
    )
    u = utmp + Sim["Timestep"] * source
    return u


def ksStepETDRK4(u, Sim):
    g = Sim["g"]
    E = Sim["E"]
    E_2 = Sim["E_2"]
    Q = Sim["Q"]
    f1 = Sim["f1"]
    f2 = Sim["f2"]
    f3 = Sim["f3"]

    v = np.fft.fft(u, axis=0)

    Nv = g * np.fft.fft(np.real(np.fft.ifft(v, axis=0)) ** 2, axis=0)
    a = E_2 * v + Q * Nv
    Na = g * np.fft.fft(np.real(np.fft.ifft(a, axis=0)) ** 2, axis=0)
    b = E_2 * v + Q * Na
    Nb = g * np.fft.fft(np.real(np.fft.ifft(b, axis=0)) ** 2, axis=0)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = g * np.fft.fft(np.real(np.fft.ifft(c, axis=0)) ** 2, axis=0)
    v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

    u = np.real(np.fft.ifft(v, axis=0))

    return u

def ks_dissRate(u, Sim):

    v = np.fft.fft(u, axis=0)
    gradv = 1j * Sim['k'] * v
    gradu = np.real(np.fft.ifft(gradv, axis=0))
    #gradu = np.gradient(u, Sim["x"], axis=0, edge_order=2)
    dissRate = Sim["SrcCoeff"] * gradu**2 
    u += Sim["Timestep"] * dissRate

    return u


def bnn_dissRate(u, Sim):
    # Compute gradient.
    v = np.fft.fft(u, axis=0)
    gradv = 1j * Sim['k'] * v
    gradu = np.real(np.fft.ifft(gradv, axis=0))

    # Compute dissipation rate.
    tf.random.set_seed(Sim["rseed"])
    x = np.concatenate([u, gradu], axis=1) / np.array([Sim["SrcFCScale"], Sim["SrcChiCFScale"]])
    dissRate = Sim["SrcCoeff"] * Sim["Model"](x)
    u += Sim["Timestep"] * dissRate

    return u


def mean_bnn_dissRate(u, Sim):
    # Compute gradient.
    v = np.fft.fft(u, axis=0)
    gradv = 1j * Sim['k'] * v
    gradu = np.real(np.fft.ifft(gradv, axis=0))

    # Compute dissipation rate using conditional mean.
    x = np.concatenate([u, gradu], axis=1) / np.array([Sim["SrcFCScale"], Sim["SrcChiCFScale"]])
    _, preds_mean, _ = compute_predictions(Sim["Model"], x, num_samples=Sim["SrcNSamples"])
    dissRate = Sim["SrcCoeff"] * np.array(preds_mean)[..., np.newaxis]
    u += Sim["Timestep"] * dissRate

    return u


def no_src(u, Sim):
    return u

