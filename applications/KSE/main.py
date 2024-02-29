import os
import sys
import argparse

import mluqprop_applications.KSE.util.data as data
from prettyPlot.parser import parse_input_file
import numpy as np
import mluqprop_applications.KSE.util.postProc as postProc
import mluqprop_applications.KSE.util.simulation as simulation

def main(args):
    # ~~~~ Init
    # Parse input
    inpt = parse_input_file(args.inputfname)

    # Initialize random seed
    np.random.seed(seed=42)

    # Initialize Simulation details
    Sim = data.simSetUp(inpt)

    # Set TF random seed for model calls during simulation.
    if Sim["SrcType"]=="bnn":
        Sim["rseed"] = args.rseed

    # ~~~~ Main
    # Run Simulation
    Result = simulation.simRun(Sim)

    # Save result to file
    if args.save_rseed:
        np.savez(os.path.join(args.outdir, args.outputfname+str(args.rseed)),
            tt=Result['tt'], uu=Result['uu'], k=Sim['k']
        )
    else:
        np.savez(os.path.join(args.outdir, args.outputfname),
            tt=Result['tt'], uu=Result['uu'], k=Sim['k']
        )

    # ~~~~ Post process
    # Plot
    if args.postProc:
        postProc.postProc(Result, Sim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Neural Network example on 1D dataset")

    parser.add_argument("--rseed", type=int, default=0)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--inputfname", default="input_src")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--postProc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outputfname", default="kse")
    parser.add_argument("--save-rseed", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if args.verbose:
        print(args)

    # ######## Run Script ########
    main(args)
