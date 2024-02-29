import numpy as np
from applications.KSE.util.integrator import *

import sys
import os
from mluqprop.BNN.util.models import load_model

def simSetUp(inpt):
    Sim = {}

    Ndof = int(inpt["Ndof"])
    Timestep = float(inpt["Timestep"])
    Tf = float(inpt["Tf"])
    Sim["Simulation name"] = inpt["Simulation name"]
    Sim["Ndof"] = Ndof
    Sim["Timestep"] = Timestep
    Sim["Tf"] = Tf

    # Post proc
    Sim["Plot"] = inpt["Plot"] == "True"
    if inpt["Simulation name"] == "KS":
        # Source term 
        try:
            Sim["SrcType"] = inpt["SrcType"]
            Sim["SrcCoeff"] = float(inpt["SrcCoeff"])
            if Sim["SrcType"]=="dissRate":
                print("Using a resolved dissipation rate term.")
                Sim["srcFunc"] = ks_dissRate
            elif Sim["SrcType"]=="bnn":
                print("Using a BNN dissipation rate term.")
                D_H = int(inpt["SrcDH"])
                D_X = int(inpt["SrcDX"])
                N_H = int(inpt["SrcNH"])
                activation = inpt["SrcActivation"]
                modelPath = inpt["SrcPath"]
                Sim["Model"] = load_model(modelPath,
                    N_H=N_H,
                    D_H=D_H,
                    D_X=D_X,
                    D_Y=1,
                    posterior_model="mvn",
                    model_type="epi",
                    activation_fn=activation)
                Sim["SrcFCScale"] = float(inpt["SrcFCScale"])
                Sim["SrcChiCFScale"] = float(inpt["SrcChiCFScale"])
                Sim["srcFunc"] = bnn_dissRate
            elif Sim["SrcType"]=="mean_bnn":
                print("Using the mean BNN dissipation rate term.")
                D_H = int(inpt["SrcDH"])
                D_X = int(inpt["SrcDX"])
                N_H = int(inpt["SrcNH"])
                activation = inpt["SrcActivation"]
                modelPath = inpt["SrcPath"]
                Sim["Model"] = load_model(modelPath,
                    N_H=N_H,
                    D_H=D_H,
                    D_X=D_X,
                    D_Y=1,
                    posterior_model="mvn",
                    model_type="epi",
                    activation_fn=activation)
                Sim["SrcFCScale"] = float(inpt["SrcFCScale"])
                Sim["SrcChiCFScale"] = float(inpt["SrcChiCFScale"])
                Sim["SrcNSamples"] = int(inpt["SrcNSamples"])
                Sim["srcFunc"] = mean_bnn_dissRate
            else:
               print("WARNING: Source term not implemented, will not use any")
               Sim["SrcType"] = None
               Sim["srcFunc"] = no_src
        except KeyError:
            Sim["SrcType"] = None
            Sim["srcFunc"] = no_src
        
        # scalars for ETDRK4
        h = Timestep
        Sim["Lx/pi"] = float(inpt["Lx/pi"])
        k = np.transpose(
            np.conj(
                np.concatenate(
                    (
                        np.arange(0, Ndof / 2.0),
                        np.array([0]),
                        np.arange(-Ndof / 2.0 + 1.0, 0),
                    )
                )
            )
        ) / (float(inpt["Lx/pi"]) / 2.0)
        ksorted = list(abs(k))
        ksorted.sort()
        kalias = ksorted[int(len(ksorted) * 2 / 3)]
        indexAlias = np.argwhere(abs(k) > kalias)
        L = k**2 - k**4
        E = np.exp(h * L)
        E_2 = np.exp(h * L / 2)
        M = 16
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat(
            [r], Ndof, axis=0
        )
        Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        f1 = h * np.real(
            np.mean(
                (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3,
                axis=1,
            )
        )
        f2 = h * np.real(
            np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
        )
        f3 = h * np.real(
            np.mean(
                (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3,
                axis=1,
            )
        )
        tmax = Tf
        nmax = round(tmax / h)
        g = -0.5j * k

        # Necessary data for simulations
        Sim["x"] = (
            float(inpt["Lx/pi"]) * np.pi * np.linspace(1, Ndof, Ndof) / Ndof
        )
        Sim["E"] = np.reshape(E, (Ndof, 1))
        Sim["E_2"] = np.reshape(E_2, (Ndof, 1))
        Sim["Q"] = np.reshape(Q, (Ndof, 1))
        Sim["f1"] = np.reshape(f1, (Ndof, 1))
        Sim["f2"] = np.reshape(f2, (Ndof, 1))
        Sim["f3"] = np.reshape(f3, (Ndof, 1))
        Sim["nmax"] = nmax
        Sim["nplt"] = 1
        Sim["g"] = np.reshape(g, (Ndof, 1))
        Sim["k"] = np.reshape(k, (Ndof, 1))
        Sim["indexAlias"] = indexAlias
        Sim["epsilon_init"] = float(inpt["epsilon_init"])

        # forward step and qoi
        Sim["stepFunc"] = ksStepETDRK4


        # Initial conditions
        ICType = inpt["ICType"]
        if ICType == "file":
            fileNameIC = inpt["fileNameIC"]
            Sim["u0"] = np.load(fileNameIC)
        elif ICType == "default":
            x = Sim["x"]
            Sim["u0"] = np.cos(x / 16) * (1 + np.sin(x / 16))
        else:
            print("IC type not recognized")

    return Sim
