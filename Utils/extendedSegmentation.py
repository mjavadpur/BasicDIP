import numpy as np
import cv2
import matplotlib.pyplot as plt


def canny(image, min_val:int = 100, max_val:int = 200):
    imgShape = image.shape
    grayscale = image
    if len(imgShape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, min_val, max_val)
    
    return edges


def sobel(image, dx:int = 1, dy:int = 1, GaussianBlurKsize:int = 3):
    imgShape = image.shape
    grayscale = image
    if len(imgShape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    
    src = cv2.GaussianBlur(image, (GaussianBlurKsize, GaussianBlurKsize), 0)

    scale = 1
    delta = 0
    ddepth = cv2.CV_64F #cv2.CV_16S
    gray = src
    # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    print('grad_x', grad_x.shape)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)


    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return edges

def laplacian(image):
    imgShape = image.shape
    grayscale = image
    if len(imgShape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Laplacian edge detection
    edges = cv2.Laplacian(grayscale, cv2.CV_8U)
    
    return edges

def prewitt(image):
    imgShape = image.shape
    grayscale = image
    if len(imgShape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    edges_x = cv2.filter2D(grayscale, -1, kernel_x)
    edges_y = cv2.filter2D(grayscale, -1, kernel_y)
    edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)
    
    return edges

def region_seg(image , thresholdFactor = 0.01):
    
    img = np.copy(image)
    imgShape = image.shape
    gray = image
    if len(imgShape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply morphological operations to remove noise and fill holes
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, thresholdFactor*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Apply the watershed algorithm to segment the image
    markers = cv2.connectedComponents(sure_fg)[1]
    markers += 1
    markers[unknown==255] = 0
    
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert the image to the correct data type, if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    markers = cv2.watershed(img, markers)

    # Apply color mapping to visualize the segmentation
    colors = np.random.randint(0, 255, (len(np.unique(markers)), 3))
    color_map = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        color_map[markers == i+1] = c
        
    return color_map

def region_segmentation(image, seed_point, connectivity=4, Sobel_dx = 1, Sobel_dy = 1, Sobel_ksize = 5):
    '''
    
    does not work right
    '''
    
    # grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elevation_map = canny(image) #cv2.Sobel(grayscale, cv2.CV_8U, Sobel_dx, Sobel_dy, ksize=Sobel_ksize)
    # elevation_map = sobel(image)
    coins = np.copy(image)

    # coin_shape = coins.shape

    # if len(coin_shape) == 3:
    #     coins = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    
    coin_gray = coins
    if coins.shape[-1] == 3:
        coin_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    markers = np.zeros_like(coin_gray, np.uint8)

    
    markers[coin_gray < 30] = 1

    markers[coin_gray > 150] = 2

    print('elevation_map', elevation_map.shape)
    print('elevation_map', elevation_map.dtype)
    print('markers', markers.shape)
    print('markers', markers.dtype)
    
    if elevation_map.shape[-1] != 3:
        elevation_map = cv2.cvtColor(elevation_map, cv2.COLOR_GRAY2BGR)

    # Convert the image to the correct data type, if needed
    if elevation_map.dtype != np.uint8:
        elevation_map = elevation_map.astype(np.uint8)
        
        
    print('elevation_map:', elevation_map.dtype, elevation_map.shape)
    print('markers:', markers.dtype, markers.shape)
    segmentation = cv2.watershed(elevation_map, markers)


    h, w = segmentation.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    segmentation = cv2.floodFill(segmentation , mask, (0,0), 255);#(segmentation - 1)
    # segmentation = ndi.binary_fill_holes(segmentation - 1)

    

    output = cv2.connectedComponentsWithStats(
        segmentation, coins, cv2.CV_32S)
    (numLabels, labeled_coins, stats, centroids) = output
    
    # labeled_coins, _ = ndi.label(segmentation)

    label_range = np.linspace(0, 1, 256)
    lut = np.uint8(plt.cm.viridis(label_range)[:,2::-1]*256).reshape(256, 1, 3) # replace viridis with a matplotlib colormap of your choice
    image_label_overlay = cv2.LUT(cv2.merge((labeled_coins, labeled_coins, labeled_coins)), lut)


    # image_label_overlay = label2rgb(labeled_coins, image=coins)
    return image_label_overlay
