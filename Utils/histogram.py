import cv2
from skimage.exposure import match_histograms

def apply_histogram_equalization(image):
    """
    Applies histogram equalization to the image.
    """   
    
    # تبدیل فضای رنگی از BGR به HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # تطابق هیستوگرام فضای رنگی H
    hsv_img[:,:,0] = cv2.equalizeHist(hsv_img[:,:,0])

    # تبدیل فضای رنگی از HSV به BGR
    matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)         
    return matched_img#cv2.equalizeHist(image)

def match_histogram(image, target_image):
    """
    Matches the histogram of the image to the histogram of the target image.
    :param target_image: Image whose histogram will be used as a reference.
    """        

    
    matched_img = match_histograms(image, target_image, channel_axis=0)
    
    return matched_img


def applyCLAHE(image, clipLimit: int = 5):
    '''
    The declaration of CLAHE
    
    @Params
    -------------
    image: input image
    clipLimit: int -> Threshold for contrast limiting
    '''
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=clipLimit)
    final_img = clahe.apply(image_bw) + 30
    
    return final_img
    
       