import sys
import time
import numpy as np
from mluqprop.BNN.util.models import (
    compute_epi_predictions_from_raw,
    compute_raw_epi_predictions,
    compute_prob_predictions,
)


def make_pred(model, Xtest, ptiles, dataFolder, npredict):
    sys.path.append(dataFolder)
    from scaling_upgraded import inv_scale_inpt, inv_scale_otpt_lrm

    start = time.time()
    predicted, aleatory = compute_raw_epi_predictions(
        model, Xtest, num_samples=npredict
    )
    (
        epipreds_mean,
        alestd,
        epistd,
        epi_ptile,
    ) = compute_epi_predictions_from_raw(predicted, aleatory, ptiles=ptiles)
    epipreds_mean = epipreds_mean[:, np.newaxis]
    alestd = alestd[:, np.newaxis]
    epistd = epistd[:, np.newaxis]
    _, _, preds_ptile = compute_prob_predictions(
        model,
        Xtest,
        num_samples=npredict,
        ptiles=ptiles,
        lowmem=True,
    )
    print(f"BNN prediction time: {time.time() - start:.2f} seconds.")

    start = time.time()
    for i in range(predicted.shape[0]):
        predicted[i, :] = inv_scale_otpt_lrm(
            predicted[i, :][:, np.newaxis],
            Xtest,
        )[:, 0]

    fullpredsphysical = np.mean(predicted, axis=0)
    fullpredsphysical_ptile = np.percentile(predicted, ptiles, axis=0)
    fullpredptilephysicallo = fullpredsphysical_ptile[0, :]
    fullpredptilephysicalhi = fullpredsphysical_ptile[1, :]
    epistdphysical = np.std(predicted, axis=0)
    alestdphysical = inv_scale_otpt_lrm(
        alestd, Xtest, #dataFolder=dataFolder
    )
    print(f"Rescaling BNN prediction time: {time.time() - start:.2f} seconds.")

    return fullpredsphysical, epistdphysical
