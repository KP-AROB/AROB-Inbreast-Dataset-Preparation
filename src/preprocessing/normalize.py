import numpy as np


def truncate_normalization(img, mask):
    """
    Normalize an image within a given ROI mask
    @img : numpy array image
    @mask : numpy array roi
    return: numpy array of the normalized image
    """
    Pmin = np.percentile(img[mask != 0], 5)
    Pmax = np.percentile(img[mask != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[mask == 0] = 0
    return np.array(normalized * 255, dtype=np.uint8)
