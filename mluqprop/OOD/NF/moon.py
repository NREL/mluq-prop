import os
import sys

import numpy as np
import torch

# import matplotlib.pyplot as plt


sys.path.append("utils")
import time

import myparser
import parallel as par
import sampler
import utils
from dataUtils import prepareData
from myProgressBar import printProgressBar
from plotFun import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inpt = myparser.parseInputFile()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Baseline data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=10000, noise=0.08)
dim = 2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nSamplesOOD = 1000

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Downsample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_flow = X
nFullData = X.shape[0]

# Create the normalizing flow
flow = sampler.createFlow(dim, inpt)
# flow = flow.to(device)
n_params = utils.get_num_parameters(flow)
par.printRoot(
    "There are {} trainable parameters in this model.".format(n_params)
)
# Train (happens on 1 proc)
flow_nll_loss = sampler.trainFlow(data_flow, flow, inpt)
sampler.checkLoss(flow_iter, flow_nll_loss)

# Evaluate probability: This is the expensive step (happens on multi processors)
log_density_np_ = sampler.evalLogProb(
    flow, data_to_downsample_, nFullData, inpt
)

par.printRoot("TRAIN ITER " + str(flow_iter))
