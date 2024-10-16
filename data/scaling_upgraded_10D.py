import pickle
import numpy as np
import os
from mluqprop import MLUQPROP_DATA_DIR as DATA_DIR

def get_scaler_inpt(scaler):
    if scaler is None:
        with open(os.path.join(DATA_DIR, "scalerInput_upgraded_10D.pkl"), "rb") as handle:
            scaler = pickle.load(handle)
    return scaler


def get_scaler_otpt(scaler):
    if scaler is None:
        with open(os.path.join(DATA_DIR, "scalerOutput_upgraded.pkl"), "rb") as handle:
            scaler = pickle.load(handle)
    return scaler


def computeLRM_fromUnscaled(
    data, max_diss_rate_unscaled=4.955625499920389, clip=True, scalerOtpt=None
):
    scalerOtpt = get_scaler_otpt(scalerOtpt)
    alpha = data[:, 3]
    beta = data[:, 4]
    gamma = data[:, 5]
    cvar = data[:, 1]
    lrm = (
        0.32
        * np.sqrt(alpha**2 + beta**2 + gamma**2)
        * np.clip(cvar, a_min=0, a_max=None)
    )
    lrm_scaled = scalerOtpt.transform(np.reshape(lrm, (-1, 1)))
    if clip:
        lrm_scaled = clipLRM_fromScaled(lrm_scaled, max_diss_rate_unscaled, scalerOtpt)
    return lrm_scaled


def computeLRM_fromScaled(
    data,
    scalerInpt=None,
    scalerOtpt=None,
    max_diss_rate_unscaled=4.955625499920389,
    clip=True,
):
    scalerInpt = get_scaler_inpt(scalerInpt)
    scalerOtpt = get_scaler_otpt(scalerOtpt)
    data_unscaled = scalerInpt.inverse_transform(data)
    lrm_scaled = computeLRM_fromUnscaled(data_unscaled, scalerOtpt=scalerOtpt)
    if clip:
        lrm_scaled = clipLRM_fromScaled(lrm_scaled, max_diss_rate_unscaled, scalerOtpt)
    return lrm_scaled


def clipLRM_fromUnscaled(lrm, max_diss_rate_unscaled, scalerOtpt=None):
    scalerOtpt = get_scaler_otpt(scalerOtpt)
    lrm_clipped = np.clip(lrm, a_min=0, a_max=1.5 * max_diss_rate_unscaled)
    lrm_clipped_scaled = scalerOtpt.transform(np.reshape(lrm_clipped, (-1, 1)))
    return lrm_clipped_scaled


def clipLRM_fromScaled(lrm, max_diss_rate_unscaled, scalerOtpt=None):
    scalerOtpt = get_scaler_otpt(scalerOtpt)
    lrm_unscaled = scalerOtpt.inverse_transform(lrm)
    lrm_clipped_scaled = clipLRM_fromUnscaled(
        lrm_unscaled, max_diss_rate_unscaled, scalerOtpt
    )
    return lrm_clipped_scaled


def scale_inpt(data, scaler=None):
    """Scale features for training

    Parameters
    ----------
    data : np.array
        Unscaled features
    scaler : RobustScaler | None
        Feature scaler utility

    Returns
    -------
    out : np.array
        scaled features
    """
    scaler = get_scaler_inpt(scaler)
    assert data.shape[1] == 10
    return scaler.transform(data)


def inv_scale_inpt(data, scaler=None):
    """Unscale features to physical space

    Parameters
    ----------
    data : np.array
        Scaled features
    scaler : RobustScaler | None
        Feature scaler utility

    Returns
    -------
    out : np.array
        unscaled features
    """
    scaler = get_scaler_inpt(scaler)
    assert data.shape[1] == 10
    return scaler.inverse_transform(data)


def scale_otpt(data, scaler=None):
    """Scale labels for training

    Parameters
    ----------
    data : np.array
        Unscaled labels
    scaler : MinMaxScaler | None
        Label scaler utility

    Returns
    -------
    out : np.array
        scaled labels
    """
    scaler = get_scaler_otpt(scaler)
    assert data.shape[1] == 1
    return scaler.transform(data)

def inv_scale_otpt(data, scaler=None):
    """Unscale labels to physical space

    Parameters
    ----------
    data : np.array
        Unscaled labels
    scaler : MinMaxScaler | None
        Label scaler utility

    Returns
    -------
    out : np.array
        unscaled labels
    """
    scaler = get_scaler_otpt(scaler)
    assert data.shape[1] == 1
    return scaler.inverse_transform(data)


def inv_scale_otpt_lrm(dataOtpt, dataInpt, scalerInpt=None, scalerOtpt=None):
    """Unscale labels predicted with LRM rescaling to physical space

    Parameters
    ----------
    dataOtpt : np.array
        Scaled labels with LRM scaling
    dataInpt : np.array
        Scaled features
    scalerInpt : RobustScaler | None
        Feature scaler utility
    scalerOtpt : MinMaxScaler | None
        Label scaler utility

    Returns
    -------
    out : np.array
        unscaled labels
    """
    lrm_scaled = computeLRM_fromScaled(
        dataInpt, scalerInpt=scalerInpt, scalerOtpt=scalerOtpt
    )
    dataOtpt = dataOtpt + lrm_scaled
    return inv_scale_otpt(dataOtpt, scalerOtpt)

def inv_scale_uncertainty(uncertainty):
    """Unscale uncertainty"

    Parameters
    ----------
    uncertainty : np.array
        uncertainty of the output of the BNN
    Returns
    -------
    out : np.array
        uncertainty in physical space
    """
    scaler = get_scaler_otpt(None)
    return uncertainty / scaler.scale_

def inv_scale_otpt_stdn(
    dataOtpt,
    mean_diss_rate=3.4006905956885154,
    std_diss_rate=0.7863628389167407,
    scalerOtpt=None,
):
    """Unscale labels predicted with std normal rescaling to physical space

    Parameters
    ----------
    dataOtpt : np.array
        Scaled labels with std normal scaling
    scalerOtpt : MinMaxScaler | None
        Label scaler utility
    mean_diss_rate : float
        Mean label in original scaled data
    std_diss_rate : float
        Std label in original scaled data
    Returns
    -------
    out : np.array
        unscaled labels
    """
    dataOtpt *= std_diss_rate
    dataOtpt += mean_diss_rate
    return inv_scale_otpt(dataOtpt, scalerOtpt)


def testInpt():
    scalerInpt_prop = np.load("scaling_input_prop_10D.npz")
    transf_ex = scale_inpt(scalerInpt_prop["input_trans_ex"])
    inv_transf_ex = inv_scale_inpt(scalerInpt_prop["input_inv_trans_ex"])
    err_transf = np.linalg.norm(transf_ex - scalerInpt_prop["transf_ex"])
    err_transf_inv = np.linalg.norm(inv_transf_ex - scalerInpt_prop["inv_transf_ex"])
    print("Error transf input:", err_transf)
    print("Error inv transf input:", err_transf_inv)


def testOtpt():
    scalerOtpt_prop = np.load("scaling_output_prop.npz")
    transf_ex = scale_otpt(scalerOtpt_prop["input_trans_ex"])
    inv_transf_ex = inv_scale_otpt(scalerOtpt_prop["input_inv_trans_ex"])
    err_transf = np.linalg.norm(transf_ex - scalerOtpt_prop["transf_ex"])
    err_transf_inv = np.linalg.norm(inv_transf_ex - scalerOtpt_prop["inv_transf_ex"])
    print("Error transf output:", err_transf)
    print("Error inv transf output:", err_transf_inv)


if __name__ == "__main__":
    testInpt()
    testOtpt()
