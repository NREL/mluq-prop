#!/usr/bin/env python

"""
Useful metrics for the training and evaluation of Bayesian neural networks.
"""

import sys
import warnings
from typing import Tuple

import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance
from welford import Welford

from mluqprop.BNN.util.models import compute_prob_predictions, compute_epi_predictions

# ==============================================================================
def weighted_norm(x:np.array, w:np.array)->float:
    """Thin wrapper for a weighted l2 norm.
    
    Params:
    x (np.array): Array of values.
    w (np.array): Array of weights.
    """
    return np.linalg.norm(np.sqrt(w) * x)


# ==============================================================================
# Frechet Inception Distance.
def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance#As_a_distance_between_probability_distributions_(the_FID_score)
    
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the mean for generated samples.
    -- mu2   : The sample mean.
    -- sigma1: The covariance matrix for generated samples.
    -- sigma2: The covariance matrix precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.

    Implementation from:
    https://arxiv.org/pdf/1706.08500.pdf
    https://github.com/bioinf-jku/TTUR
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# ==============================================================================
# Compute the signal-to-noise ratio.
def compute_snr(model)->np.array:
    """Compute the signal-to-noise ratio (in dB) of Bayesian neural network model weights.

    Args:
        model: Bayesian neural network model.

    Returns:
        (np.array): Signal-to-noise ratio for all weights in dB.
        
    Note: when loading in a previously trained model, be sure to load from weights, not using model.load() as that method does not support variational models.
    """
    # Dummy input to query the posterior.
    dummy = np.array([[0.]])
    
    # Loop through the layers, grab info from posterior.
    mus = []
    stds = []
    for i in range(len(model.layers)-1):
        mus.append(model.layers[i]._posterior(dummy).mean().numpy())
        stds.append(model.layers[i]._posterior(dummy).stddev().numpy())

    # Return the SNR in dB.
    return 20.*np.log10(np.divide(np.abs(np.concatenate(mus)), np.concatenate(stds)))


# ==============================================================================
# Compute the signal-to-noise ratio for Flipout models.
def compute_snr_flipout(model)->np.array:
    """Compute the signal-to-noise ratio (in dB) of Bayesian neural network model weights.

    Args:
        model: Bayesian neural network model.

    Returns:
        (np.array): Signal-to-noise ratio for all weights in dB.
        
    Note: this function is for use with Flipout layers.
    Note: when loading in a previously trained model, be sure to load from weights, not using model.load() as that method does not support variational models.
    """
    
    # Loop through the layers, grab info from posterior.
    mus = []
    stds = []
    for i in range(len(model.layers)-1):
        mus.append(model.layers[i].kernel_posterior.mean().numpy().flatten())
        stds.append(model.layers[i].kernel_posterior.stddev().numpy().flatten())

    # Return the SNR in dB.
    return 20.*np.log10(np.divide(np.abs(np.concatenate(mus)), np.concatenate(stds)))


# ==============================================================================
# Wasserstein distance.
def wass1d(y_true, y_pred):
    """Thin wrapper to SciPy's wasserstein distance function for implementation as custom metric in TensorFlow/Keras.

    Args:
        y_true: True labels.
        y_pred: Predicted labels

    Returns:
        1D Wasserstein Distance.
    """
    return wasserstein_distance(y_true.squeeze(), y_pred.squeeze())


# ========================================================================
# Streaming model moments.
def compute_model_moments_online(model, x:np.ndarray, nalea:int=100, nepi:int=200, sampling=False)->Tuple[float, float, float]:
    """Vectorized version of Bayesian neural network prediction statistics on a dataset. Computations are done in a streaming (online) fashion with Welford's algorithm.

    Args:
        model: Bayesian neural network model.
        x (np.ndarray): Data points to test on.
        nalea (int, optional): Number of aleatoric predictions. Defaults to 100.
        nepi (int, optional): Number of model forms to sample. Defaults to 200.
        sampling (bool, optional): Whether or not to use sampling for aleatoric uncertainty. Defaults to False.

    Returns:
        Tuple[float, float, float]: Predictive mean, Epistemic uncertainty, Aleatoric uncertainty.
    """
    
    # preds = np.zeros((x.shape[0], nalea, nepi))  # for testing

    # Initialize welford objects.
    if sampling:
        wm = Welford()
        wepi = Welford()
        for i in range(nepi):
            # Reshaping and repeating the input for vectorized prediction.
            y = np.repeat(np.reshape(x, (x.shape[0], x.shape[1], 1)), nalea, axis=2)
            z = np.reshape(y.transpose(0,2,1), (-1, x.shape[1]))
            
            # Sample predictions.
            v = model(z).sample()
            vv = np.reshape(v, (x.shape[0], nalea))  # some reshaping to make things easier.
            # preds[:, :, i] = vv  # for testing
            wm.add_all(np.transpose(vv))
            wepi.add(np.mean(vv, axis=1))
    else:
        wmu = Welford()
        wsigma = Welford()
        for i in range(nepi):
            v = model(x)
            wmu.add(v.mean().numpy().squeeze())
            wsigma.add(v.stddev().numpy().squeeze())
    
    if sampling:
        # Mean prediction at each inducing point.    
        pmean = wm.mean
        # Aleatoric uncertainty is the mean of the standard deviation of the predictions from each model.
        pale = np.sqrt(wm.var_s)
        # Epistemic uncertainty is the standard deviation of the mean prediction from each model.
        pepi = np.sqrt(wepi.var_s)
    else:
        pmean = wmu.mean[:, np.newaxis]
        pale = wsigma.mean[:, np.newaxis]
        pepi = np.sqrt(wmu.var_s)[:, np.newaxis]  # using sample variance estimator
        # pepi = np.sqrt(wmu.var_p)[:, np.newaxis]  # using population variance estimator
    
    return pmean, pepi, pale


# ========================================================================
# Vectorized moment distance.
def vec_moment_distance(model, uips, nalea:int=100, nepi:int=200)->Tuple[float, float]:
    """Vectorized version of moment distance calculation for Bayesian neural network prediction on the inducing points.

    Args:
        model: Bayesian neural network model.
        uips: Inducing points.
        nalea (int, optional): Number of aleatoric predictions. Defaults to 100.
        nepi (int, optional): Number of model forms to sample. Defaults to 200.

    Returns:
        Tuple[float, float]: L2 norm of the difference of mean predictions, L2 norm of the difference of aleatoric uncertainty.
    """
    x = uips["loc"]
    
    # Allocate space to store predictions.
    preds = np.zeros((x.shape[0], nalea, nepi))
    mean = np.zeros(uips["mean"].shape)
    ale = np.zeros(uips["aleat"].shape)
    
    # Reshaping and repeating the input for vectorized prediction.
    y = np.repeat(np.reshape(x, (x.shape[0], x.shape[1], 1)), nalea, axis=2)
    z = np.reshape(y.transpose(0,2,1), (-1, x.shape[1]))
    
    # Sampling models from the posterior.
    for i in range(nepi):
        # v = model.predict(z)
        v = model(z).sample()
        preds[:, :, i] = np.reshape(v, (x.shape[0], nalea))
    
    # Mean prediction at each inducing point.    
    pmean = np.mean(preds, axis=(1, 2))
    
    # Epistemic uncertainty is the standard deviation of the mean prediction from each model.
    pepi = np.std(np.mean(preds, axis=1), axis=1)
    
    # Aleatoric uncertainty is the mean of the standard deviation of the predictions from each model.
    pale = np.mean(np.std(preds, axis=1),axis=1)

    return np.linalg.norm(uips["mean"] - pmean, ord=2), np.linalg.norm(uips["aleat"] - pale, ord=2)


# ========================================================================
# Compute model predictions with moments in vectorized manner.
def vec_model_moments(model, x:np.array, nalea:int=100, nepi:int=200):
    """Vectorized version of moment distance calculation for Bayesian neural network prediction on the inducing points.

    Args:
        model: Bayesian neural network model.
        x (np.array): Data points to test on.
        nalea (int, optional): Number of aleatoric predictions. Defaults to 100.
        nepi (int, optional): Number of model forms to sample. Defaults to 200.

    Returns:
        preds: Predictions.
        mean: Mean predictions.
        pepi: Epistemic uncertainty.
        pale: Aleatoric uncertainty.
    """

    # Allocate space to store predictions.
    preds = np.zeros((x.shape[0], nalea, nepi))
    
    # Reshaping and repeating the input for vectorized prediction.
    y = np.repeat(np.reshape(x, (x.shape[0], x.shape[1], 1)), nalea, axis=2)
    z = np.reshape(y.transpose(0,2,1), (-1, x.shape[1]))
    
    # Sampling models from the posterior.
    for i in range(nepi):
        # v = model.predict(z)
        v = model(z).sample()
        preds[:, :, i] = np.reshape(v, (x.shape[0], nalea))
    
    # Mean prediction at each inducing point.    
    pmean = np.mean(preds, axis=(1, 2))
    
    # Epistemic uncertainty is the standard deviation of the mean prediction from each model.
    pepi = np.std(np.mean(preds, axis=1), axis=1)
    
    # Aleatoric uncertainty is the mean of the standard deviation of the predictions from each model.
    pale = np.mean(np.std(preds, axis=1),axis=1)

    return preds, pmean, pepi, pale


# ========================================================================
# Summarize the uncertainties.
def summarize_uncertainty(model, xpt: np.array, nalea:int=100, nepi:int=100)->Tuple[np.array, float, float, float]:
    """Summarize Bayesian neural network prediction uncertaintites.

    Args:
        model: Bayesian neural network.
        xpt: Evaluation point.
        nalea (int, optional): Number of aleatoric predictions. Defaults to 100.
        nepi (int, optional): Number of model forms to sample. Defaults to 100.

    Returns:
        Tuple[np.array, float, float, float]: Predictions, mean prediction, epistemic uncertainty, aleatoric uncertainty.
    """
    x = np.repeat(np.array([xpt]), nalea, axis=0)
    
    predicted = compute_prob_predictions(model, x, num_samples=nepi)[0]

    pmean = np.mean(predicted)
    pepi = np.std(np.mean(predicted, axis=0))
    pale = np.mean(np.std(predicted, axis=0))

    return predicted, pmean, pepi, pale


# ========================================================================
# Moment distance.
def moment_distance(model, uips, nalea:int=100, nepi:int=100)->Tuple[float, float]:
    """Moment distance calculation for Bayesian neural network prediction on the inducing points.

    Args:
        model: Bayesian neural network model.
        uips: Inducing points.
        nalea (int, optional): Number of aleatoric predictions. Defaults to 100.
        nepi (int, optional): Number of model forms to sample. Defaults to 200.
    _summary_
    
    Returns:
        Tuple[float, float]: L2 norm of the difference of mean predictions, L2 norm of the difference of aleatoric uncertainty.
    """
    x = uips["loc"]
    
    pmean = np.zeros(uips["mean"].shape)
    pale = np.zeros(uips["aleat"].shape)
    
    for i in range(len(x)):
        predicted, pmean[i], pepi, pale[i] = summarize_uncertainty(model, x[i], nalea, nepi)

    # return mean, ale
    return np.linalg.norm(uips["mean"] - pmean, ord=2), np.linalg.norm(uips["aleat"] - pale, ord=2)


def conditionalAverage(x, y, nbin):
    try:
        assert len(x) == len(y)
    except:
        print("conditional average x and y have different dimension")
        print("dim x = ", len(x))
        print("dim y = ", len(y))
        sys.exit()
    # Bin conditional space
    mag = np.amax(x) - np.amin(x)
    x_bin = np.linspace(
        np.amin(x) - mag / (2 * nbin), np.amax(x) + mag / (2 * nbin), nbin
    )
    weight = np.zeros(nbin)
    weightVal = np.zeros(nbin)
    asum = np.zeros(nbin)
    bsum = np.zeros(nbin)
    avalsum = np.zeros(nbin)
    bvalsum = np.zeros(nbin)
    inds = np.digitize(x, x_bin)

    a = abs(y - x_bin[inds - 1])
    b = abs(y - x_bin[inds])
    c = a + b
    a = a / c
    b = b / c

    for i in range(nbin):
        asum[i] = np.sum(a[np.argwhere(inds == i)])
        bsum[i] = np.sum(b[np.argwhere(inds == i + 1)])
        avalsum[i] = np.sum(
            a[np.argwhere(inds == i)] * y[np.argwhere(inds == i)]
        )
        bvalsum[i] = np.sum(
            b[np.argwhere(inds == i + 1)] * y[np.argwhere(inds == i + 1)]
        )
    weight = asum + bsum
    weightVal = avalsum + bvalsum

    return x_bin, weightVal / (weight)


# ========================================================================
def model2model(model, refmodel, X:np.array, nepi:int=200)->Tuple[float, float, float, float]:
    """Compare a reduced BNN model to a reference BNN model.

    Args:
        model: Bayesian neural network model to be compared.
        refmodel: Bayesian neural network model to be used as the reference.
        X (np.array): Data points to test on.
        nepi (int, optional): Number of model forms to sample. Defaults to 200.

    Returns:
        Tuple[float, float, float, float]: Wasserstein distance of mean predictions, relative error in mean, relative error of aleatoric uncertainty, relative erorr of epistemic uncertainty.
    """
    
    refmean, refepi, refale = compute_model_moments_online(refmodel, X, nepi, sampling=False)
    testmean, testepi, testale = compute_model_moments_online(model, X, nepi, sampling=False)
    
    wass = wass1d(refmean, testmean)
    relmean = np.linalg.norm(refmean - testmean, ord=2) / np.linalg.norm(refmean, ord=2)
    relale = np.linalg.norm(refale - testale, ord=2) / np.linalg.norm(refale, ord=2)
    relepi = np.linalg.norm(refepi - testepi, ord=2) / np.linalg.norm(refepi, ord=2)
    
    return wass, relmean, relale, relepi
