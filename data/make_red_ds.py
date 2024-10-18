import numpy as np
import argparse

parser = argparse.ArgumentParser(description="make dataset")
parser.add_argument(
    "-n",
    "--n",
    type=int,
    metavar="",
    required=True,
    help="number of uips points",
    default=10,
)
args, unknown = parser.parse_known_args()

A=np.load("Scaled_March16_KmeansOutput_AllCase_lrmScale_10D.npz")
ind=np.load(f"downsampled_best_{args.n}.npz")["indices"]
np.savez(f"downsampled_ds_{args.n}.npz", Xtrain=A["Xtrain"][ind], Ytrain=A["Ytrain"][ind])
