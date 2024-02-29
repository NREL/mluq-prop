#!/usr/bin/env python3

import os
import shutil
import pickle
import glob
import argparse
import numpy as np

def main(args):
    
    # Make a directory for the best model.
    bestdir = os.path.join(args.resultsdir, "best/")
    os.makedirs(os.path.dirname(bestdir), exist_ok=True)
    
    # Use the log file if available.
    logpath = os.path.join(args.resultsdir, args.logfile)
    if os.path.exists(logpath):
        results = np.genfromtxt(logpath, delimiter=",")
        idx = np.nanargmin(results[:, 1])
        bestjobid = int(results[idx, 0])
        print(f"JobID of best model is: {bestjobid} with loss {results[idx, 1]}")
        
        print(f"Copying JobID {bestjobid} to {bestdir}")
        shutil.copy(os.path.join(args.resultsdir, f"hyperjob_{bestjobid}_history.pkl"), os.path.join(bestdir, "best.pkl"))
    else:
        # Otherwise, loop through all of the histories.
        print(f"WARNING: could not find log file {args.logfile}.")
        jobids = []
        bestlosses = []
        
        # Loop through all stored histories and compute the best loss for that history.
        histories = glob.glob(os.path.join(args.resultsdir, "*.pkl"))
        for f in histories:
            with open(f, "rb") as fid:
                losses = pickle.load(fid)
            bestlosses.append(np.min(losses["loss"]))
            jobids.append(int(f.split("_")[-2]))
        
        # Print out the information.
        idx = np.nanargmin(bestlosses)
        bestjobid = jobids[idx]   
        print(f"JobID of best model is: {bestjobid} with loss {bestlosses[idx]}")
        
        print(f"Copying JobID {bestjobid} to {bestdir}")
        shutil.copy(histories[idx], os.path.join(bestdir, "best.pkl"))
    
    fp = open(args.jobfile, "rb")
    for i, line in enumerate(fp):
        if i == bestjobid:
            print("Best model deployed was...")
            print(line)
            break
    fp.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess hyperparameter tuning runs.")
    
    parser.add_argument("--logfile", default="None", type=str, help="Log file containing best losses and jobids.")
    parser.add_argument("--resultsdir", type=str, help="Directory containing results files.")
    parser.add_argument("--jobfile", type=str, help="Job file containing hyperparameters.")
    
    args = parser.parse_args()
    main(args)
