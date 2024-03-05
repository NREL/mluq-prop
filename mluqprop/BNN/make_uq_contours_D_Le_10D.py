import numpy as np
from prettyPlot.plotting import pretty_multi_contour, plt
import os
import sys
from mluqprop import MLUQPROP_DATA_DIR as DATA_DIR
sys.path.append(DATA_DIR)
from scaling_upgraded_10D import inv_scale_otpt_lrm, inv_scale_otpt, inv_scale_inpt, scale_inpt, computeLRM_fromScaled, inv_scale_otpt_stdn, inv_scale_uncertainty, scale_otpt

from mluqprop.BNN.util.metrics import compute_snr, compute_snr_flipout, vec_model_moments, conditionalAverage, weighted_norm
from mluqprop.BNN.util.models import load_model, compute_prob_predictions, compute_raw_epi_predictions, compute_epi_predictions_from_raw
from mluqprop.BNN.util.dns import dns_data_loader, dns_partial_data_loader
from mluqprop.BNN.util.input_parser import parse_input_deck

import argparse

def plot_dns(field_3d, xmin=400, xmax=800):
    fig = plt.figure()
    shape = field_3d.shape
    print(shape)
    assert len(shape)==3
    mid = shape[2] // 2   
    plt.imshow(np.transpose(field_3d[xmin:xmax,:,mid]))
    plt.colorbar()
    plt.show()

def plot_2d(field_2d):
    fig = plt.figure()
    shape = field_2d.shape
    print(shape)
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

# Load the model in.
model = load_model(
    fpath=os.path.join(simparams.checkpath, "best/", "best"),
    D_X=D_X,
    D_H=simparams.hidden_dim,
    D_Y=DY,
    N_H=simparams.num_layers,
    kl_weight=1 / N if simparams.model_type == "variational" else 1.,
    model_type=simparams.model_type,
    activation_fn=simparams.nonlin,
    posterior_model=simparams.posterior_model,
    split=simparams.split
)

# Print the model summary.
print(f"Model loaded from {simparams.checkpath}.")
model.summary()


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
    print(f"Feature {i}")
    print(f"\tDATASET SCALED min {min_dat:.2f} max {max_dat:.2f}")
    print(f"\tD_LE FW4 min {min_ble_4:.2f} max {max_ble_4:.2f}")
    print(f"\tD_LE FW16 min {min_ble_16:.2f} max {max_ble_16:.2f}")
print(f"output")
min_dat = np.amin(dataset_output_unscaled)
max_dat = np.amax(dataset_output_unscaled)
min_ble_4 = np.amin(otpt_fw4/chi_lam)
max_ble_4 = np.amax(otpt_fw4/chi_lam)
min_ble_16 = np.amin(otpt_fw16/chi_lam)
max_ble_16 = np.amax(otpt_fw16/chi_lam)
print(f"\tDATASET UNSCALED min {min_dat:.2f} max {max_dat:.2f}")
print(f"\tD_LE UNSCALED FW4 min {min_ble_4:.2f} max {max_ble_4:.2f}")
print(f"\tD_LE UNSCALED FW16 min {min_ble_16:.2f} max {max_ble_16:.2f}")


#  we need the array of predictions to compute the epistemic uncertainty in physical space
predicted_fw4, aleatory_fw4 = compute_raw_epi_predictions(
        model, inpt_fw4_scaled, num_samples=args.npredict
    )
predicted_fw16, aleatory_fw16 = compute_raw_epi_predictions(
        model, inpt_fw16_scaled, num_samples=args.npredict
    )
# compute the mean, uncertainties (in data space), and the percentiles (in data space)
epipreds_mean_fw4, alestd_fw4, epistd_fw4, _ = compute_epi_predictions_from_raw(
    predicted_fw4, aleatory_fw4,
)
epipreds_mean_fw4 = epipreds_mean_fw4[:, np.newaxis]
alestd_fw4 = alestd_fw4[:, np.newaxis]
epistd_fw4 = epistd_fw4[:, np.newaxis]

epipreds_mean_fw4 = inv_scale_otpt_lrm(epipreds_mean_fw4, inpt_fw4_scaled)
epistd_fw4 = inv_scale_uncertainty(epistd_fw4)
alestd_fw4 = inv_scale_uncertainty(alestd_fw4)

epipreds_mean_fw16, alestd_fw16, epistd_fw16, _ = compute_epi_predictions_from_raw(
    predicted_fw16, aleatory_fw16,
)
epipreds_mean_fw16 = epipreds_mean_fw16[:, np.newaxis]
alestd_fw16 = alestd_fw16[:, np.newaxis]
epistd_fw16 = epistd_fw16[:, np.newaxis]

epipreds_mean_fw16 = inv_scale_otpt_lrm(epipreds_mean_fw16, inpt_fw16_scaled)
epistd_fw16 = inv_scale_uncertainty(epistd_fw16)
alestd_fw16 = inv_scale_uncertainty(alestd_fw16)


#plot_dns(data_fw4["chiSFS"])
#plot_dns(data_fw16["chiSFS"])

pred_fw4 = np.reshape(epipreds_mean_fw4, (x_len, y_len))
pred_fw16 = np.reshape(epipreds_mean_fw16, (x_len, y_len))
#plot_2d(pred_fw4)
#plot_2d(pred_fw16)


x = np.linspace(0, 2.56300000e-02, x_tot_len+1)
y = np.linspace(0, 2.33000000e-03, y_len + 1)
z = np.linspace(0, 2.33000000e-03, z_len + 1)


os.makedirs("Figures", exist_ok = True)
os.makedirs(os.path.join("Figures", "D_Le_10D"), exist_ok = True)
figureFolder = os.path.join("Figures", "D_Le_10D")

pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [np.reshape(otpt_fw4 / chi_lam, (x_len, y_len)), pred_fw4],
                     xbound=[x[x_min], x[x_max - 1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["True", "BNN"],
                     listCBLabel=[r"$\chi_{\rm SFS}$ / $\chi_{\rm lam} [-]$", r"$\chi_{\rm SFS}$ / $\chi_{\rm lam} [-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     log_scale_list=[True, True],
                     vminList=[0.01, 0.01],
                     vmaxList=[2.1, 2.1],
                     #globalTitle="FW4",
                     globalTitle=None,
                     grid=False,
                     #log_scale_list=[True, True],
                    ) 
plt.savefig(os.path.join(figureFolder, "chiSFS_D_Le_fw4.png"))


pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [np.reshape(otpt_fw16 /chi_lam, (x_len, y_len)), pred_fw16],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["True", "BNN"],
                     listCBLabel=[r"$\chi_{\rm SFS}$ / $\chi_{\rm lam} [-]$", r"$\chi_{\rm SFS}$ / $\chi_{\rm lam} [-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     log_scale_list=[True, True],
                     vminList=[0.01, 0.01],
                     vmaxList=[2.6, 2.6],
                     #globalTitle="FW16",
                     globalTitle=None,
                     grid=False,
                     #log_scale_list=[True, True],
                    ) 
plt.savefig(os.path.join(figureFolder, "chiSFS_D_Le_fw16.png"))

pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [(np.reshape(otpt_fw4, (x_len, y_len)) + np.reshape(inpt_fw4[:,2] * chi_lam,(x_len, y_len)))/chi_lam , np.clip(pred_fw4 * chi_lam + np.reshape(inpt_fw4[:,2] * chi_lam,(x_len, y_len)), a_min=0.01, a_max=None)/chi_lam],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["True", "BNN"],
                     listCBLabel=[r"$\widetilde{\chi}$ / $\chi_{\rm lam} [-]$",  r"$\widetilde{\chi}$ / $\chi_{\rm lam} [-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW4",
                     globalTitle=None,
                     grid=False,
                     vminList=[0.01, 0.01],
                     vmaxList=[15, 15],
                     log_scale_list=[True, True],
                     display_cbar_list=[False, True],
                     #figsize=(6,4),
                     cbar_pad=0.1,
                     cbar_size_percent=5,
                    )
plt.subplots_adjust(wspace=0.4)
plt.savefig(os.path.join(figureFolder, "chiF_D_Le_fw4.png"))
pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [(np.reshape(otpt_fw16, (x_len, y_len)) + np.reshape(inpt_fw16[:,2] * chi_lam,(x_len, y_len)))/chi_lam, np.clip(pred_fw16 * chi_lam + np.reshape(inpt_fw16[:,2] * chi_lam,(x_len, y_len)), a_min=0.01, a_max=None)/chi_lam],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["True", "BNN"],
                     listCBLabel=[r"$\widetilde{\chi}$ / $\chi_{\rm lam} [-]$",  r"$\widetilde{\chi}$ / $\chi_{\rm lam} [-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW16",
                     globalTitle=None,
                     grid=False,
                     vminList = [0.01, 0.01],
                     vmaxList = [15, 15],
                     log_scale_list=[True, True],
                     display_cbar_list=[False, True],
                     cbar_pad=0.1,
                     cbar_size_percent=5,
                    )
plt.subplots_adjust(wspace=0.4)
plt.savefig(os.path.join(figureFolder, "chiF_D_Le_fw16.png"))
pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [np.reshape(epistd_fw4, (x_len, y_len)), np.reshape(alestd_fw4,(x_len, y_len))],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["Epistemic", "Aleatoric"],
                     listCBLabel=["", "$[-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW4",
                     globalTitle=None,
                     grid=False,
                     cbar_pad=0.1,
                     cbar_size_percent=5,
                    )
plt.subplots_adjust(wspace=0.4)
plt.savefig(os.path.join(figureFolder, "unc_D_Le_fw4.png"))
pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [np.reshape(epistd_fw16, (x_len, y_len)), np.reshape(alestd_fw16,(x_len, y_len))],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["Epistemic", "Aleatoric"],
                     listCBLabel=["", "$[-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW16",
                     globalTitle=None,
                     grid=False,
                     cbar_pad=0.1,
                     cbar_size_percent=5,
                    )
plt.subplots_adjust(wspace=0.4)
plt.savefig(os.path.join(figureFolder, "unc_D_Le_fw16.png"))
pretty_multi_contour(
                     [x[x_min:x_max]],
                     [np.reshape(alestd_fw4,(x_len, y_len))/ np.reshape(epistd_fw4, (x_len, y_len))],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["Aleatoric/Epistemic"],
                     listCBLabel=["$[-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW4",
                     globalTitle=None,
                     grid=False,
                     cbar_pad=0.1,
                     cbar_size_percent=5,
                    )
plt.savefig(os.path.join(figureFolder, "ratio_unc_D_Le_fw4.png"))
pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [np.reshape(epistd_fw16, (x_len, y_len)), np.reshape(alestd_fw16,(x_len, y_len))],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["Epistemic", "Aleatoric"],
                     listCBLabel=["", "$[-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW16",
                     globalTitle=None,
                     grid=False,
                     cbar_pad=0.1,
                     cbar_size_percent=5,
                    )
plt.subplots_adjust(wspace=0.4)
plt.savefig(os.path.join(figureFolder, "unc_D_Le_fw16.png"))
pretty_multi_contour(
                     [x[x_min:x_max]],
                     [np.reshape(alestd_fw16,(x_len, y_len))/ np.reshape(epistd_fw16, (x_len, y_len))],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["Aleatoric/Epistemic"],
                     listCBLabel=["$[-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW4",
                     globalTitle=None,
                     grid=False,
                     cbar_pad=0.1,
                     cbar_size_percent=5,
                    )
plt.savefig(os.path.join(figureFolder, "ratio_unc_D_Le_fw16.png"))
pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [np.reshape(epistd_fw4, (x_len, y_len)), np.reshape(alestd_fw4,(x_len, y_len))],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["Epistemic", "Aleatoric"],
                     listCBLabel=["", "$[-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW4",
                     globalTitle=None,
                     log_scale_list=[True, True],
                     grid=False
                    ) 
plt.savefig(os.path.join(figureFolder, "log_unc_D_Le_fw4.png"))
pretty_multi_contour(
                     [x[x_min:x_max], x[x_min:x_max]],
                     [np.reshape(epistd_fw16, (x_len, y_len)), np.reshape(alestd_fw16,(x_len, y_len))],
                     xbound=[x[x_min], x[x_max-1]],
                     ybound=[y[0], y[-1]],
                     listTitle=["Epistemic", "Aleatoric"],
                     listCBLabel=["", "$[-]$"],
                     #listXAxisName=["x [m]", "x [m]"], 
                     #listYAxisName=["y [m]", ""],
                     listXAxisName=["", ""], 
                     listYAxisName=["", ""],
                     #globalTitle="FW16",
                     globalTitle=None,
                     log_scale_list=[True, True],
                     grid=False
                    ) 
plt.savefig(os.path.join(figureFolder, "log_unc_D_Le_fw16.png"))

plt.show()

