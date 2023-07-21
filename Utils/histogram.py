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
    
    #  # تبدیل فضای رنگی از BGR به HSV
    # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_target_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

    # # تطابق هیستوگرام فضای رنگی H
    # hsv_img[:,:,0] = cv2.equalizeHist(hsv_img[:,:,0], hsv_target_img[:,:,0])

    # # تبدیل فضای رنگی از HSV به BGR
    # matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    #  # تبدیل فضای رنگی از BGR به HSV
    # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_target_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

    # # تطابق هیستوگرام فضای رنگی H
    # hsv_img[:,:,0] = cv2.calcHist(hsv_img[:,:,0])
    # cv.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    # hsv_target_img[:,:,0] = cv2.equalizeHist(hsv_target_img[:,:,0])

    # # تبدیل فضای رنگی از HSV به BGR
    # matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    matched_img = match_histograms(image, target_image, channel_axis=0)
    
    return matched_img
        