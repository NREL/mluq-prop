#!/usr/bin/env python

"""
Script to generate launcher job file for dimension reduction with variable number of DenseVariational layers.
"""

import argparse
import itertools

def main(args):
    # Generate list of layer masks.
    layer_masks = list(map(list, itertools.product([0, 1], repeat=args.num_layers)))
    layer_masks.pop(0)  # remove the all-zero mask.
    layer_masks.pop(-1) # remove the all-one mask.

    # Write parameters to file for launching jobs.
    f = open(args.filename, "w")
    for i, mask in enumerate(layer_masks):
        f.write(f"python dimreduceTrainer.py --input {args.input} --ckptepoch --no-metrics --filename dimreduce_{i} --checkpath $CHECKPATH/dimreduce_{i} --verbose --layer_mask {' '.join(str(i) for i in mask)} --refmodel_path {args.refmodel_path} --npredict {args.npredict} > $LOGDIR/dimreducejob_{i}.log\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write hyperparameters to file for dimension reduction layer mask sweeps.")
    
    parser.add_argument("--filename", type=str, default="dimreducejobs.txt", help="Filename to write the jobs to.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of hidden layers.")
    parser.add_argument("--input", type=str, default="inputs/dimreduce.in")
    parser.add_argument("--refmodel_path", type=str, default="/projects/mluq/mluq-prop/models/experiments/bestLRM/best/best/best")
    parser.add_argument("--npredict", help="Number of predictions for model comparison.", type=int, default=50)
    
    args = parser.parse_args()
    main(args)
