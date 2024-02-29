import numpy as np


def rescaleData(np_data):
    minVal = np.amin(np_data, axis=0)
    maxVal = np.amax(np_data, axis=0)

    np_data_rescaled = np_data.copy()
    np_data_rescaled = (np_data_rescaled - minVal) / (
        0.125 * (maxVal - minVal)
    ) - 4
    return np_data_rescaled, minVal, maxVal


def unrescaleData(np_data, minVal, maxVal):
    np_data_rescaled = np_data.copy()

    np_data_rescaled = (np_data + 4) * 0.125 * (maxVal - minVal) + minVal

    return np_data_rescaled
