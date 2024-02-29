#!/usr/bin/env python3

"""
Routines for comparing model output with DNS data.
================================
credit to Shashank Y. for originally developing these codes.
"""

import os
import pickle
import time
from datetime import timedelta

import numpy as np
from scipy import stats

from sklearn.metrics import mean_squared_error

import joblib


# ========================================================================
# Data loader for .npz DNS data files
def dns_partial_data_loader(fpath:str):
    """Load downsampled DNS data (for use with KSE).

    Args:
        fpath (str): Path to data file.

    Returns:
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.
    """
    tmp = np.load(fpath, allow_pickle=True)
    tmp = tmp[tmp.files[0]]
    x = tmp.item()['x']
    y = tmp.item()['y']
    return x, y


# ========================================================================
# Data loader for .npz DNS data files
def dns_data_loader(fpath:str):
    """Load DNS data.

    Args:
        fpath (str): Path to data file.

    Returns:
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.
        xtest (np.ndarray): Test input data.
        ytest (np.ndarray): Test output data.
    """
    data = np.load(fpath, allow_pickle=True)
    x = data['Xtrain']
    y = data['Ytrain']
    xtest = data['Xtest']
    ytest = data['Ytest']
    
    return x, y, xtest, ytest


# ========================================================================
# Data loader for extrapolatory uncertainty DNS data files produced by Malik H.
def load_extrap_data(fpath_orig, fpath_extrap, nextrap:int=-1):
    """Load in the original and extrapolatory data.

    Args:
        fpath_orig (str): Path to original dataset.
        fpath_extrap (str): Path to extrapolatory dataset.
        nextrap (int, optional): How much extrapolatory data to inclue. Defaults to -1 (whole dataset). Only should be used for data generated with normalizing flow.

    Returns:
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.
        xtest (np.ndarray): Test input data.
        ytest (np.ndarray): Test output data.
    """
    orig = np.load(fpath_orig, allow_pickle=True)
    extrap = np.load(fpath_extrap, allow_pickle=True)
    
    xextrap = extrap['backgroundData'][0:nextrap]
    yextrap = extrap['labels'][0:nextrap]
    
    x = np.concatenate((orig['Xtrain'], xextrap))
    y = np.concatenate((orig['Ytrain'], yextrap))
    xtest = orig['Xtest']
    ytest = orig['Ytest']
    return x, y, xtest, ytest, xextrap, yextrap


"""
Shashank's post-processing routines.
# ========================================================================
# Function to post-process results with model.
def predictENNResidualAllCase(model, InpFolderList, scalerInputFile,
                              ResultFolderList, kaDeltaFile, scaleOutputFile=None):

    start = time.time()

    Inp = np.loadtxt(InpFolderList, dtype=np.str)
    OutputDir = np.loadtxt(ResultFolderList, dtype=np.str)
    LamChi, Cburnt, Diff = np.loadtxt(kaDeltaFile, unpack=True)

    filter = [2, 4, 8, 16, 32]

    # Scale the input data
    scalerInput = joblib.load(scalerInputFile)

    if scaleOutputFile is not None:
        scalerOutput = joblib.load(scaleOutputFile)

    nameInp = ['FC_', 'FCvar_', 'ChicF_', 'alpha_', 'beta_', 'gamma_', 'GradAlphaGradC_', 'GradBetaGradC_', 'GradGammaGradC_', 'FD_', 'GradTGradC_']
    
    Predict = {}
    DNSdata = {}
    for iif, folder in enumerate(Inp):
        print("Currently working on {0}".format(folder))
        r2filename = os.path.abspath(os.path.join(OutputDir[iif],
                                                  'rsquare.txt'))
        r2file = open(r2filename, 'w')
        scalingCoeff = [Cburnt[iif], (Cburnt[iif]*(1.0-Cburnt[iif])), LamChi[iif], LamChi[iif], LamChi[iif], LamChi[iif], 1.0, 1.0, 1.0, Diff[iif], 1.0]

        for delta in filter:
            print("Currently working on filter width {0}".format(delta))

            fileName = "Filtered_Input_"+str(delta)+'.pkl'
            fname = os.path.abspath(os.path.join(folder, fileName))
            file = open(fname, 'rb')
            input = pickle.load(file)
            file.close()

            fileName = "Filtered_Output_"+str(delta)+'.pkl'
            fname = os.path.abspath(os.path.join(folder, fileName))
            file = open(fname, 'rb')
            TrainOut = pickle.load(file)
            file.close()

            numInput = len(nameInp)
            nx = input[list(input.keys())[0]].shape[0]
            ny = input[list(input.keys())[0]].shape[1]
            nz = input[list(input.keys())[0]].shape[2]
            InpArr = np.zeros((nx*ny*nz, numInput))
            CFilt = np.zeros(nx*ny*nz)

            for i, key in enumerate(nameInp):
                InpArr[:, i] = np.reshape(input[key+str(delta)], (nx*ny*nz))

            for i, key in enumerate(nameInp):
                InpArr[:, i] = InpArr[:, i]/scalingCoeff[i]

            InpArr = scalerInput.transform(InpArr)

            name = "ChiC_"+str(delta)
            resName = "residual_"+str(delta)
            DNSdata[name] = TrainOut[name].flatten(order='C')/LamChi[iif]
            DNSdata[resName] = TrainOut[resName].flatten(order='C')/LamChi[iif]
            InpArr = Variable(torch.from_numpy(InpArr)).float()

            sgs, xc = model.forward(InpArr)

            OutputArr = np.zeros((nx*ny*nz, 2))
            OutputArr[:, 0] = sgs.detach().numpy().flatten(order='C')
            OutputArr[:, 1] = xc.detach().numpy().flatten(order='C')
            
            Predict[resName] = scalerOutput.inverse_transform(OutputArr[:, 0:1]).flatten(order='C')
            Predict[name] = OutputArr[:, 1]
            
            print("residual max predict {0} and DNS {1}".format(Predict[resName].max(), DNSdata[resName].max()))
            print("chiC max predict {0} and DNS {1}".format(Predict[name].max(), DNSdata[name].max()))
            print("residual min predict {0} and DNS {1}".format(Predict[resName].min(), DNSdata[resName].min()))
            print("chiC min predict {0} and DNS {1}".format(Predict[name].min(), DNSdata[name].min()))

            CFilt = input['FC_'+str(delta)].flatten(order='C')
            nbins = 32

            MeansDNS, binEdgesDNS, nbinDNS = stats.binned_statistic(CFilt, DNSdata[resName],
                                                                    statistic='mean',
                                                                    bins=nbins)
        
            MeansPredict, binEdgesPredict, nbinPredict = stats.binned_statistic(CFilt,
                                                                                Predict[resName],
                                                                                statistic='mean',
                                                                                bins=nbins)

            bin_width = (binEdgesDNS[1] - binEdgesDNS[0])
            bin_centers = binEdgesDNS[1:] - bin_width/2.0

            # Output the data for plotting
            filename = os.path.join(OutputDir[iif], 'ConditionalStats_residual_'+str(delta)+'.npz')
            np.savez_compressed(filename, xbins=bin_centers,
                                PredictMean=MeansPredict,
                                DNSMean=MeansDNS)

            MeansDNS, binEdgesDNS, nbinDNS = stats.binned_statistic(CFilt, DNSdata[name],
                                                                    statistic='mean',
                                                                    bins=nbins)
        
            MeansPredict, binEdgesPredict, nbinPredict = stats.binned_statistic(CFilt,
                                                                                Predict[name],
                                                                                statistic='mean',
                                                                                bins=nbins)

            bin_width = (binEdgesDNS[1] - binEdgesDNS[0])
            bin_centers = binEdgesDNS[1:] - bin_width/2.0

            # Output the data for plotting
            filename = os.path.join(OutputDir[iif], 'ConditionalStats_ChiC_'+str(delta)+'.npz')
            np.savez_compressed(filename, xbins=bin_centers,
                                PredictMean=MeansPredict,
                                DNSMean=MeansDNS)
            
            # rsq = np.sqrt(mean_squared_error(DNSdata[name], Predict[name])/np.mean(DNSdata[name]*DNSdata[name]))
            # rsqSgs = np.sqrt(mean_squared_error(DNSdata[resName], Predict[resName])/np.mean(DNSdata[resName]*DNSdata[resName]))

            rsq = ErrorMetric(Predict[name], DNSdata[name], 0.01)
            rsqSgs = ErrorMetric(Predict[resName], DNSdata[resName], 0.01)
            
            r2file.write("{} \t {} \t {} \n".format(delta, rsq, rsqSgs))
            print("RMSE error for ChiC filter {0} is {1}".format(delta, rsq))
            print("RMSE error for Chisgs filter {0} is {1}".format(delta, rsqSgs))

        r2file.close()
        fileName = os.path.abspath(os.path.join(OutputDir[iif],
                                                'Prediction.pkl'))
        with open(fileName, 'wb') as handle:
            pickle.dump(Predict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fileName = os.path.abspath(os.path.join(OutputDir[iif],
                                                'FilterDNS.pkl'))
        with open(fileName, 'wb') as handle:
            pickle.dump(DNSdata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.time() - start
    print("Elapsed time " + str(timedelta(seconds=end)) +
          " (or {0:f} seconds)".format(end))


def ErrorMetric(DNN, DNS, thresh):
    maxDNS = np.max(np.abs(DNS))
    threshold = thresh*maxDNS

    DNSclip = DNS[np.abs(DNS)>threshold]
    DNNclip = DNN[np.abs(DNS)>threshold]

    rsq = np.sqrt(mean_squared_error(DNSclip,
                                     DNNclip)/np.mean(DNSclip*DNSclip))
    return rsq
"""
