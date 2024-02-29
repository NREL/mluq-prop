#!/usr/bin/env python

"""
Script to generate hyperparameters and write launcher job file to tune model hyperparameters.
"""

from sklearn.model_selection import ParameterSampler, ParameterGrid
import argparse

def main(args):
    # Generate the grid of parameters to search.
    param_grid = {
        "hidden_dim": [5, 10, 15, 20],
        "num_layers": [2, 3, 4],
        "batch_size": [256, 512, 1024, 2048, 4096, 8192, 16384],
        "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
    }
    params = list(ParameterGrid(param_grid))

    # Write parameters to file for launching jobs.
    f = open(args.filename, "w")
    for i, param in enumerate(params):
        f.write(f"python trainer.py --input inputs/hypertune.in --hidden_dim {param['hidden_dim']} --num_layers {param['num_layers']} --batch_size {param['batch_size']} --learning_rate {param['learning_rate']} --ckptepoch --no-metrics --filename hyperjob_{i} --checkpath $CHECKPATH/hyperjob_{i} --tuning --verbose > $LOGDIR/hyperjob_{i}.log\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write hyperparameters to file for hyperparameter tuning.")
    
    parser.add_argument("--filename", type=str, default="hyperjobs.txt", help="Filename to write hyperparameters to.")
    
    args = parser.parse_args()
    main(args)