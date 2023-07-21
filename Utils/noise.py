import numpy as np
import cv2
            

def add_gauss_noise(image, mean = 0, var = 0.1):
    '''
    Add Gaussian noise to the image.

    Parameters
    ----------
    mean : float, optional
        Mean of the Gaussian distribution to generate noise (default is 0).
    var : float, optional
        Variance of the Gaussian distribution to generate noise (default is 0.1).
    '''
    sigma = var**0.5
    
    
    gauss = np.random.normal(mean,sigma,image.size)
    gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(image,gauss)

    
    return img_gauss

def add_sp_noise(image, s_vs_p :float = 0.5, amount: float = 0.004):
    '''
    Add Salt & Pepper noise to the image.

    Parameters
    ----------
    s_vs_p : float, optional
        Ratio of salt to pepper (default is 0.5).
    amount : float, optional
        Overall proportion of image pixels to replace with noise (default is 0.004).
    '''
    out = np.copy(image)

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
    coords = tuple(map(lambda dim: np.random.randint(0, dim, num_salt), image.shape))
    out[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
    coords = tuple(map(lambda dim: np.random.randint(0, dim, num_pepper), image.shape))
    out[coords] = 0

    return out

def add_poisson_noise(image, **kwargs):
    '''
    Add Poisson noise to the image.

    The noise is added as per a Poisson distribution. This function does not take any additional parameters.
    '''
    # Convert the image to double data type
    image = image.astype(np.float64)

    # Scale the image to the range of 0-1
    image /= np.max(image)

    # Convert the image to represent counts in the range of 0-255
    image *= 255

    # Apply the Poisson noise
    noisy = np.random.poisson(image)

    # Normalize the noisy image
    noisy = noisy / np.max(noisy)
    
    noisy *= 255.
    noisy = noisy.astype(np.uint8) 

    return noisy



def add_speckle_noise(image, mean: float = 0, var: float = 0.5):
    """
    Add Speckle noise to the image.

    Speckle noise is a multiplicative noise. This function does not take any additional parameters.
    """
    
    sigma = var**0.5
    
    noise = np.random.normal(mean, sigma, image.shape)

    noise = noise.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
    noise = image + image * noise

    return noise

