import numpy as np
import cv2

def apply_threshold(image, threshold_value):
    """
    Applies a binary threshold to the image.
    :param threshold_value: Threshold value for binarization.
    """
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_gamma_correction(image, gamma):
    """
    Applies gamma correction to the image.
    :param gamma: Gamma value for correction.
    """
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return gamma_corrected

def apply_negative(image):
    """
    Applies negative transformation to the image.
    """
    return 255 - image

def apply_log_transform(image, c=1):
    """
    Applies logarithmic transformation to the image.
    :param c: Constant for scaling the transformation.
    """
    log_transformed = c * np.log(1 + image)
    
    log_transformed = np.array(log_transformed, dtype='uint8')
    
    minVal = np.min(log_transformed)
    maxVal = np.max(log_transformed)
    
    log_transformed = (log_transformed-minVal)/(maxVal-minVal) * 255
    log_transformed = np.array(log_transformed, dtype='uint8')
    return log_transformed

def apply_contrast_stretching(image, r_min=0, r_max=255):
    """
    Applies contrast stretching to the image.
    :param r_min: Minimum intensity value after stretching.
    :param r_max: Maximum intensity value after stretching.
    """
    stretched = np.interp(image, [np.min(image), np.max(image)], [r_min, r_max])
    return np.array(stretched, dtype='uint8')
