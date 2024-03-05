import numpy as np
from prettyPlot.plotting import pretty_multi_contour, plt
from prettyPlot.progressBar import print_progress_bar
import os
import sys
from mluqprop import MLUQPROP_DATA_DIR as DATA_DIR
sys.path.append(DATA_DIR)
from scaling_upgraded_10D import inv_scale_otpt_lrm, inv_scale_otpt, inv_scale_inpt, scale_inpt, computeLRM_fromScaled, inv_scale_otpt_stdn, inv_scale_uncertainty, scale_otpt

from mluqprop.BNN.util.metrics import compute_snr, compute_snr_flipout, vec_model_moments, conditionalAverage, weighted_norm
from mluqprop.BNN.util.models import load_model, compute_prob_predictions, compute_raw_epi_predictions, compute_epi_predictions_from_raw
from mluqprop.BNN.util.dns import dns_data_loader, dns_partial_data_loader
from mluqprop.BNN.util.input_parser import parse_input_deck
import mluqprop.BNN.util.parallel as par

import argparse

def plot_dns(field_3d, xmin=400, xmax=800):
    fig = plt.figure()
    shape = field_3d.shape
    par.printRoot(shape)
    assert len(shape)==3
    mid = shape[2] // 2   
    plt.imshow(np.transpose(field_3d[xmin:xmax,:,mid]))
    plt.colorbar()
    plt.show()

def plot_2d(field_2d):
    fig = plt.figure()
    shape = field_2d.shape
    par.printRoot(shape)
    assert len(shape)==2
    plt.imshow(np.transpose(field_2d))
    plt.colorbar()
    plt.show()

def make_input(struct, xmin=400, xmax=800, ymin=0, ymax=128, zmin=64, zmax=65):
    size = np.ndarray.flatten(struct["FC"][xmin:xmax,ymin:ymax,zmin:zmax]).shape[0]
    input_array = np.zeros((size,10))
    output_array = np.zeros((size,1))
    input_array[:,0] = np.ndarray.flatten(struct["FC"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,1] = np.ndarray.flatten(struct["FCvar"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,2] = np.ndarray.flatten(struct["ChicF"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,3] = np.ndarray.flatten(struct["alpha"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,4] = np.ndarray.flatten(struct["beta"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,5] = np.ndarray.flatten(struct["gamma"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,6] = np.ndarray.flatten(struct["GradAlphaGradC"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,7] = np.ndarray.flatten(struct["GradBetaGradC"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,8] = np.ndarray.flatten(struct["GradGammaGradC"][xmin:xmax,ymin:ymax,zmin:zmax])
    input_array[:,9] = np.ndarray.flatten(struct["FD"][xmin:xmax,ymin:ymax,zmin:zmax])
    #input_array[:,10] = np.ndarray.flatten(struct["GradTGradC"][xmin:xmax,ymin:ymax,zmin:zmax])
    #input_array[:,11] = np.ndarray.flatten(struct["FOmegaC"][xmin:xmax,ymin:ymax,zmin:zmax])
    output_array[:,0] = np.ndarray.flatten(struct["chiSFS"][xmin:xmax,ymin:ymax,zmin:zmax])
    return input_array, output_array 


parser = argparse.ArgumentParser(description="Evaluate BNN model performance on DNS data.")

parser.add_argument("--input", help="Input deck.")
parser.add_argument("--npredict", type=int, default=100)
parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True, help="Rescale data to physical space.")
parser.add_argument("--errorbars", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to include error bars on the inducing points plots.")
parser.add_argument("--condmean", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to compute the conditional means.")
parser.add_argument("--datatype", default="lrm", choices=["lrm", "dns", "normalized"], help="The type of data being used in evaluation, for scaling purposes.")
parser.add_argument("--all_condmeans", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to compute the conditional means for all variables.")

args = parser.parse_args()

# Parse the input deck.
if args.input is None:
    raise NameError("No input deck provided. Please specify with: --input <input_deck>")
simparams = parse_input_deck(args.input)

if args.rescale:
    simparams.filename = simparams.filename + "_rescaled"

# Load DNS data.
simparams.use_lean = True if simparams.use_lean == "True" else False  # manually handle lean data flag.
simparams.split = True if simparams.split == "True" else False  # manually handle the split flag.
if simparams.use_lean:
    Xtrain, Ytrain = dns_partial_data_loader(simparams.data_fpath)
    Xtest = Xtrain
    Ytest = Ytrain
else:
    Xtrain, Ytrain, Xtest, Ytest = dns_data_loader(simparams.data_fpath)
    freq = 1
    Xtrain = Xtrain[::freq]
    Ytrain = Ytrain[::freq]
    Xtest = Xtest[::freq]
    Ytest = Ytest[::freq]
N, D_X = Xtrain.shape
DY = Ytrain.shape[1]

x_min=400
x_max=1000
x_len = x_max - x_min

maxC = 0.24194361400000003
diff = 0.0001547382159334776
chi_lam = 184.61529339640268
rhoOmega = 1744.4185

data_fw4 = np.load(os.path.join(DATA_DIR, "D_Le_fw4.npz"))
data_fw16 = np.load(os.path.join(DATA_DIR, "D_Le_fw16.npz"))

feat=list(data_fw4.keys())

x_tot_len = data_fw4[feat[0]].shape[0]
y_len = data_fw4[feat[0]].shape[1]
z_len = data_fw4[feat[0]].shape[2]
z_mid = data_fw4[feat[0]].shape[2]//2 

#plot_dns(data_fw4["chiSFS"])

inpt_fw4, otpt_fw4 = make_input(data_fw4, xmin=x_min, xmax=x_max, ymin=0, ymax=y_len, zmin=z_mid, zmax=z_mid+1)
inpt_fw16, otpt_fw16 = make_input(data_fw16, xmin=x_min, xmax=x_max, ymin=0, ymax=y_len, zmin=z_mid, zmax=z_mid+1)
inpt_fw4[:,0] = inpt_fw4[:,0]/maxC
inpt_fw16[:,0] = inpt_fw16[:,0]/maxC
inpt_fw4[:,1] = inpt_fw4[:,1]/(maxC*(1-maxC))
inpt_fw16[:,1] = inpt_fw16[:,1]/(maxC*(1-maxC))
inpt_fw4[:,2] = inpt_fw4[:,2]/chi_lam
inpt_fw16[:,2] = inpt_fw16[:,2]/chi_lam
inpt_fw4[:,3] = inpt_fw4[:,3]/chi_lam
inpt_fw16[:,3] = inpt_fw16[:,3]/chi_lam
inpt_fw4[:,4] = inpt_fw4[:,4]/chi_lam
inpt_fw16[:,4] = inpt_fw16[:,4]/chi_lam
inpt_fw4[:,5] = inpt_fw4[:,5]/chi_lam
inpt_fw16[:,5] = inpt_fw16[:,5]/chi_lam
inpt_fw4[:,9] = inpt_fw4[:,9]/diff
inpt_fw16[:,9] = inpt_fw16[:,9]/diff

#inpt_fw4[:,11] = inpt_fw4[:,11]/rhoOmega
#inpt_fw16[:,11] = inpt_fw16[:,11]/rhoOmega
#inpt_fw4[:,11] = inpt_fw4[:,11]/rhoOmega
#inpt_fw16[:,11] = inpt_fw16[:,11]/rhoOmega
#otpt_fw4 = otpt_fw4/chi_lam
#otpt_fw16 = otpt_fw16/chi_lam

inpt_fw4_scaled =  scale_inpt(inpt_fw4)
inpt_fw16_scaled =  scale_inpt(inpt_fw16)
otpt_fw4_scaled =  scale_otpt(otpt_fw4)
otpt_fw16_scaled =  scale_otpt(otpt_fw16)

#inpt_fw4_scaled_old = np.load("orig_scaler/D_Le_fw4_input_scaleold.npy")
#inpt_fw16_scaled_old = np.load("orig_scaler/D_Le_fw16_input_scaleold.npy")
#otpt_fw4_scaled_old = np.load("orig_scaler/D_Le_fw4_output_scaleold.npy")
#otpt_fw16_scaled_old = np.load("orig_scaler/D_Le_fw16_output_scaleold.npy")


dataset = np.load(os.path.join(DATA_DIR, "Scaled_March16_KmeansOutput_AllCase_lrmScale_10D.npz"))
dataset_output_unscaled = inv_scale_otpt_lrm(dataset["Ytrain"], dataset["Xtrain"])

for i in range(10):
    min_dat = np.amin(dataset["Xtrain"][:,i])
    max_dat = np.amax(dataset["Xtrain"][:,i])
    min_ble_4 = np.amin(inpt_fw4_scaled[:,i])
    max_ble_4 = np.amax(inpt_fw4_scaled[:,i])
    min_ble_16 = np.amin(inpt_fw16_scaled[:,i])
    max_ble_16 = np.amax(inpt_fw16_scaled[:,i])
    par.printRoot(f"Feature {i}")
    par.printRoot(f"\tDATASET SCALED min {min_dat:.2f} max {max_dat:.2f}")
    par.printRoot(f"\tD_LE FW4 min {min_ble_4:.2f} max {max_ble_4:.2f}")
    par.printRoot(f"\tD_LE FW16 min {min_ble_16:.2f} max {max_ble_16:.2f}")
par.printRoot(f"output")
min_dat = np.amin(dataset_output_unscaled)
max_dat = np.amax(dataset_output_unscaled)
min_ble_4 = np.amin(otpt_fw4/chi_lam)
max_ble_4 = np.amax(otpt_fw4/chi_lam)
min_ble_16 = np.amin(otpt_fw16/chi_lam)
max_ble_16 = np.amax(otpt_fw16/chi_lam)
par.printRoot(f"\tDATASET UNSCALED min {min_dat:.2f} max {max_dat:.2f}")
par.printRoot(f"\tD_LE UNSCALED FW4 min {min_ble_4:.2f} max {max_ble_4:.2f}")
par.printRoot(f"\tD_LE UNSCALED FW16 min {min_ble_16:.2f} max {max_ble_16:.2f}")


nSnap_, startSnap_ = par.partitionData(inpt_fw4_scaled.shape[0])

distToDS_fw4_ = np.zeros(nSnap_)
print_progress_bar(
    0,
    nSnap_,
    prefix=" %d / %d " % (0, nSnap_),
    suffix="Complete",
    length=20,
    extraCond=(par.irank == par.iroot),
)
for i in range(nSnap_):
    distToDS_fw4_[i] = np.amin(np.linalg.norm(dataset["Xtrain"] - np.reshape(inpt_fw4_scaled[startSnap_ + i],(1,-1)), axis=1))
    print_progress_bar(
        i+1,
        nSnap_,
        prefix=" %d / %d " % (i+1, nSnap_),
        suffix="Complete",
        length=20,
        extraCond=(par.irank == par.iroot),
    )
distToDS_fw4 = par.gatherNelementsInArray(
    distToDS_fw4_, inpt_fw4_scaled.shape[0] 
)
if par.irank == par.iroot:
    np.save("distToDS_D_Le_FW4.npy", distToDS_fw4)

nSnap_, startSnap_ = par.partitionData(inpt_fw16_scaled.shape[0])
distToDS_fw16_ = np.zeros(nSnap_)
print_progress_bar(
    0,
    nSnap_,
    prefix=" %d / %d " % (0, nSnap_),
    suffix="Complete",
    length=20,
    extraCond=(par.irank == par.iroot),
)
for i in range(nSnap_):
    distToDS_fw16_[i] = np.amin(np.linalg.norm(dataset["Xtrain"] - np.reshape(inpt_fw16_scaled[startSnap_ + i],(1,-1)), axis=1))
    print_progress_bar(
        i+1,
        nSnap_,
        prefix=" %d / %d " % (i+1, nSnap_),
        suffix="Complete",
        length=20,
        extraCond=(par.irank == par.iroot),
    )
distToDS_fw16 = par.gatherNelementsInArray(
    distToDS_fw16_, inpt_fw16_scaled.shape[0]
)
if par.irank == par.iroot:
    np.save("distToDS_D_Le_FW16.npy", distToDS_fw16)

