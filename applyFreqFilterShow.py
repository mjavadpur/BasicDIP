import numpy as np
import Utils.showImages as showImg
import Utils.frequencyFilter as freqFilter
from skimage.metrics import structural_similarity as ssim

def applyFreqFilterAndShowImages(image):
    
    w, h, channels = image.shape
    
    img1 = np.copy(image)
    low_pass_filter_image = freqFilter.low_pass_filter(img1, 1000)
    mssim_low_pass, grad = ssim(img1,low_pass_filter_image, full=True, win_size = 7, channel_axis=2)
    
    img2 = np.copy(image)
    high_pass_image = freqFilter.high_pass_filter(img2, 3)
    mssim_high_pass, grad = ssim(img2,high_pass_image, full=True, win_size = 7, channel_axis=2)
    
    img3 = np.copy(image)
    band_reject_image = freqFilter.band_reject_filter(img3, int(w/2)-20, int(w/2))
    mssim_band_reject, grad = ssim(img3,band_reject_image, full=True, win_size = 7, channel_axis=2)
    
    img4 = np.copy(image)
    band_pass_image = freqFilter.band_pass_filter(img4, 2, h)
    mssim_band_pass, grad = ssim(img4,band_pass_image, full=True, win_size = 7, channel_axis=2)

    img5 = np.copy(image)
    wiener_image = freqFilter.wiener_filter(img5, psf=None, k=0.8)
    mssim_wiener, grad = ssim(img5,wiener_image, full=True, win_size = 7, channel_axis=2)

    showImg.show_mult_img(2, 3,\
        [image, low_pass_filter_image, high_pass_image, band_reject_image, band_pass_image, wiener_image],\
        ['Original Image',\
            f'Low pass filter (SSIM: {mssim_low_pass:.3f})',\
            f'High pass filter (SSIM: {mssim_high_pass:.3f})',\
            f'Band reject filter (SSIM: {mssim_band_reject:.3f})',\
            f'Band pass filter (SSIM: {mssim_band_pass:.3f})',\
            f'wiener filter (SSIM: {mssim_wiener:.3f})'])