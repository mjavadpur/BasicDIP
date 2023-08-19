import cv2
import matplotlib.pyplot as plt

import Utils.showImages as showImg
import Utils.noise as noise
import Utils.filter as filter
from skimage.metrics import structural_similarity as ssim

def applyFilterOnNoisyImage(ORIGINAL_IMAGE, noisyImage):


    img_adaptive = filter.adaptive_filter(image=noisyImage)
    mssim_adaptive, grad = ssim(ORIGINAL_IMAGE,img_adaptive, full=True, win_size = 7, channel_axis=2)

    img_gaussian = filter.gaussian_filter(image=noisyImage)
    mssim_gaussian, grad = ssim(ORIGINAL_IMAGE,img_gaussian, full=True, win_size = 7, channel_axis=2)

    img_high_boost = filter.high_boost_filter(image=noisyImage)
    mssim_high_boost, grad = ssim(ORIGINAL_IMAGE,img_high_boost, full=True, win_size = 7, channel_axis=2)

    img_high_pass = filter.high_pass_filter(image=noisyImage)
    mssim_high_pass, grad = ssim(ORIGINAL_IMAGE,img_high_pass, full=True, win_size = 7, channel_axis=2)

    img_low_pass = filter.low_pass_filter(image=noisyImage)
    mssim_low_pass, grad = ssim(ORIGINAL_IMAGE,img_low_pass, full=True, win_size = 7, channel_axis=2)

    img_mean = filter.mean_filter(image=noisyImage)
    mssim_mean, grad = ssim(ORIGINAL_IMAGE,img_mean, full=True, win_size = 7, channel_axis=2)

    img_median = filter.median_filter(image=noisyImage)
    mssim_median, grad = ssim(ORIGINAL_IMAGE,img_median, full=True, win_size = 7, channel_axis=2)




    showImg.show_mult_img(3, 3, [ORIGINAL_IMAGE, noisyImage, img_adaptive, img_gaussian, img_high_boost, img_high_pass, img_low_pass, img_mean, img_median],\
        ['ORIGINAL IMAGE','Noisy Image',\
            f'Adaptive filter (SSIM: {mssim_adaptive:.3f})',\
            f'Gaussian filter (SSIM: {mssim_gaussian:.3f})',\
            f'High boost filter (SSIM: {mssim_high_boost:.3f})',\
            f'High pass filter (SSIM: {mssim_high_pass:.3f})',\
            f'Low pass filter (SSIM: {mssim_low_pass:.3f})',\
            f'Mean filter (SSIM: {mssim_mean:.3f})',\
            f'Median filter (SSIM: {mssim_median:.3f})'])