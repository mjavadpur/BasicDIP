
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

def compute_lbp(image, radius:int = 1):
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp

def apply_gabor_filters(image, frequency:float =0.2, theta: float =0.4):
    kernel_size = (31, 31)
    sigma = 5.0
    theta = 0.4
    lambd = 10.0
    gamma = 0.5
    gabor_filters = gabor(image, frequency=frequency, theta=theta)
    filtered_image = gabor_filters[0]
    return filtered_image

def compute_lpq(image, radius:int = 3):
    n_points = 8 * radius
    lpq = local_binary_pattern(image, n_points, radius, method='uniform')
    return lpq           
        