import argparse
import os
import sys
import time

import numpy as np
import torch
import uips.sampler as sampler
import uips.utils.parallel as par
from prettyPlot.parser import parse_input_file
from uips.utils.dataUtils import prepareData
from uips.utils.fileFinder import find_input
from uips.utils.plotFun import *
from uips.utils.torchutils import get_num_parameters


def prepareBackgroundData(npArr):
    # Check that dataset shape make sense
    nFullData = npArr.shape[0]
    nDim = npArr.shape[1]

    # Distribute dataset
    if par.irank == par.iroot:
        print("LOAD DATA ... ", end="")
        sys.stdout.flush()
    par.comm.Barrier()
    nSnap_, startSnap_ = par.partitionData(nFullData)
    npArr_ = npArr[startSnap_ : startSnap_ + nSnap_, :].astype("float32")
    par.printRoot("DONE!")

    # Rescale data
    if par.irank == par.iroot:
        print("RESCALE DATA ... ", end="")
        sys.stdout.flush()
    par.comm.Barrier()
    par.printRoot("DONE!")

    return npArr_


parser = argparse.ArgumentParser(description="OOD sampler")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    metavar="",
    required=False,
    help="Input file",
    default="input",
)
args, unknown = parser.parse_known_args()

inpt_file = find_input(args.input)
inpt = parse_input_file(inpt_file)
use_normalizing_flow = inpt["pdf_method"].lower() == "normalizingflow"
use_bins = inpt["pdf_method"].lower() == "bins"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parameters to save
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List of sample size
nSamples = [int(float(n)) for n in inpt["nSamples"].split()]
# Data size used to adjust the sampling probability
nWorkingDataAdjustment = int(float(inpt["nWorkingDataAdjustment"]))
if nWorkingDataAdjustment < 0:
    use_serial_adjustment = False
else:
    use_serial_adjustment = True
# Data size used to learn the data probability
nWorkingDatas = [int(float(n)) for n in inpt["nWorkingData"].split()]
if len(nWorkingDatas) == 1:
    nWorkingDatas = nWorkingDatas * int(inpt["num_pdf_iter"])
for nWorkingData in nWorkingDatas:
    if not nWorkingData in nSamples:
        nSamples += [nWorkingData]
# Do we compute the neighbor distance criterion
computeCriterion = inpt["computeDistanceCriterion"] == "True"
try:
    nSampleCriterionLimit = int(inpt["nSampleCriterionLimit"])
except:
    nSampleCriterionLimit = int(1e5)

try:
    supportSize = float(inpt["supportSize"])
except KeyError:
    supportSize = 1.2
if supportSize <= 1:
    msg = "ERROR: supportSize should be >1\n"
    msg += f"Found to be {supportSize}"
    sys.exit(msg)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
use_gpu = (
    (inpt["use_gpu"] == "True")
    and (torch.cuda.is_available())
    and (par.irank == par.iroot)
)
if use_gpu:
    # GPU SETTING
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    # CPU SETTING
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
# REPRODUCIBILITY
torch.manual_seed(int(inpt["seed"]) + par.irank)
np.random.seed(int(inpt["seed"]) + par.irank)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Prepare Data and scatter across processors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_to_downsample_, _, working_data, nFullData = prepareData(inpt)

dim = data_to_downsample_.shape[1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Get data probability
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_for_pdf_est = working_data

if use_normalizing_flow:
    # Create the normalizing flow
    flow = sampler.createFlow(dim, 0, inpt)
    n_params = get_num_parameters(flow)
    par.printRoot(
        "There are {} trainable parameters in this model.".format(n_params)
    )

    # Train (happens on 1 proc)
    flow_nll_loss = sampler.trainFlow(data_for_pdf_est, flow, 0, inpt)

    # Evaluate probability: This is the expensive step (happens on multi processors)
    log_density_np_ = sampler.evalLogProbNF(flow, data_to_downsample_, 0, inpt)
if use_bins:
    bin_pdfH, bin_pdfEdges = sampler.trainBinPDF(data_for_pdf_est, 0, inpt)
    # Evaluate probability: This is the expensive step (happens on multi processors)
    log_density_np_ = sampler.evalLogProbBIN(data_to_downsample_, 0, inpt)
    # par.printAll(f"min {np.amin(log_density_np_)} max {np.amax(log_density_np_)}")

# Get minimum of probability
try:
    min_percentile = float(inpt["min_percentile"])
    log_density_np = par.allgather1DList(log_density_np_, nFullData)
    if use_bins:
        index_keep = np.argwhere(log_density_np > np.log(1e-100))
        min_log_density = np.percentile(
            log_density_np[index_keep[:, 0]], min_percentile
        )
    else:
        min_log_density = np.percentile(log_density_np, min_percentile)
except KeyError:
    if use_bins:
        index_keep = np.argwhere(log_density_np_ > np.log(1e-100))
        min_log_density_ = np.amin(log_density_np_[index_keep[:, 0]])
    else:
        min_log_density_ = np.amin(log_density_np_)
    min_log_density = par.allminScalar(min_log_density_)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Generate background data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
minDim = np.zeros(dim)
maxDim = np.zeros(dim)
minDim_ = np.zeros(dim)
maxDim_ = np.zeros(dim)
for idim in range(dim):
    minDim_[idim] = np.amin(data_to_downsample_[:, idim])
    maxDim_[idim] = np.amax(data_to_downsample_[:, idim])
for idim in range(dim):
    minDim[idim] = par.allminScalar(minDim_[idim])
    maxDim[idim] = par.allmaxScalar(maxDim_[idim])


for idim in range(dim):
    maxDim[idim] += (maxDim[idim] - minDim[idim]) * (supportSize - 1)
    minDim[idim] -= (maxDim[idim] - minDim[idim]) * (supportSize - 1)


nBackground = int(float(inpt["nBackground"]))
backgroundData = np.zeros((nBackground, dim))
for idim in range(dim):
    backgroundData[:, idim] = np.random.uniform(
        minDim[idim], maxDim[idim], nBackground
    )

backgroundData_ = prepareBackgroundData(backgroundData)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Check probability of background data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if use_normalizing_flow:
    log_density_background_ = sampler.evalLogProbNF(
        flow, backgroundData_, 0, inpt
    )
if use_bins:
    log_density_background_ = sampler.evalLogProbBIN(backgroundData_, 0, inpt)

indexKeep = np.argwhere(log_density_background_ < min_log_density)
backgroundData_keep_ = backgroundData_[indexKeep[:, 0], :]
size_keep_ = len(indexKeep[:, 0])
size_keep = int(par.allsumScalar(size_keep_))
par.printRoot(f"Keep {size_keep*100/nBackground:.2f}% of background data")
origData = par.gatherNelementsInArray(data_to_downsample_, nFullData)
backgroundData_keep = par.gatherNelementsInArray(
    backgroundData_keep_, size_keep
)
if par.irank == par.iroot:
    np.savez(
        f"{inpt['prefixBackgroundData']}.npz",
        origData=origData,
        backgroundData=backgroundData_keep,
    )

# if par.irank==par.iroot:
#    plt.show()
