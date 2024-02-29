#!/usr/bin/env python

"""
Script to write launcher job file for extrapolatory uncertainty.
"""

import os
import argparse
import glob

def main(args):
    # Remove the old jobfile.
    OUTFILE = "extrapjobs.txt"
    if not args.append:
        try:
            os.remove(OUTFILE)
        except OSError:
            pass
    
    if args.extraptype == "nf":
        # Normalizing flows.
        datafname = args.datafname
        uipsfname = args.uipsfname
        
        for ndata in [1e4, 1e5, 1e6, 1e7]:
            # Write to jobfile.
            with open(OUTFILE, "a") as f:
                f.write(f"python extrapTrainer.py --input inputs/extrap.in --extraptype nf --extrap_data_fpath {os.path.join(args.extrap_data_path, datafname)} --nextrap {int(ndata)} --filename NF_{ndata} --extrap_uips {os.path.join(args.extrap_data_path, uipsfname)} --checkpath $CHECKPATH/NF_{ndata} --verbose --ckptepoch --no-metrics > $LOGDIR/NF_{ndata}.log\n")
    elif args.extraptype == "sbo":
        # Soft brownian offset.
        for file in glob.glob(os.path.join(args.extrap_data_path, f"{args.data_basefname}")):
            # Generate the path to the inducing points file.
            uipsfname = file.split('/')[-1].split('_')
            uipsfname[-1] = uipsfname[-1].split('.')[0]
            uipsfname.pop(-2)  # get rid of "labeled"
            uipsfname.insert(0, "frechetDistRef")
            uipsfname.insert(-1, "100")
            uipsfname = "_".join(uipsfname)
            uipsfname += ".npz"
            uipsfpath = os.path.join(args.extrap_data_path, uipsfname)
            
            name = os.path.splitext(file.split('/')[-1])[0]
            
            # Write to jobfile.
            with open(OUTFILE, "a") as f:
                f.write(f"python extrapTrainer.py --input inputs/extrap.in --extraptype sbo --extrap_data_fpath {file} --extrap_uips {uipsfpath} --filename {name} --checkpath $CHECKPATH/{name} --verbose --ckptepoch --no-metrics > $LOGDIR/{name}.log\n")
    else:
        raise ValueError(f"Extrapolation type {args.extraptype} not recognized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write launcher job file for extrapolatory uncertainty.")
    
    parser.add_argument("--extraptype", default="nf", choices=["nf", "sbo"], help="Extrapolation type. Either 'nf' for normalizing flow, or 'sbo' for Soft Brownian Offset.")
    parser.add_argument("--extrap_data_path", help="Path to directory with extrapolation data. DO NOT include the trailing '/'!")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=False, help="Append new jobs to the job file, if it exists..")
    
    # Arguments for filename conventions.
    parser.add_argument("--datafname", default="backgroundComb_highD_labeled_LRM_normalization.npz", help="Training data filename. ONLY used with --extraptype nf.")
    parser.add_argument("--data_basefname", default="*labeled_fixed*", help="Common base name for SBO data. ONLY used with --extraptype sbo.")
    parser.add_argument("--uipsfname", default="inducing_synth_100.npz", help="Inducing points filename.")
    
    args = parser.parse_args()
    main(args)
