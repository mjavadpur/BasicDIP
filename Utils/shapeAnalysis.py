import cv2
import numpy as np
import math  

def houghTransformCircle(image, method = cv2.HOUGH_GRADIENT_ALT, dp = 1, min_dist = 0,\
                    param1=200, param2=0.005, minRadius=0, maxRadius=0, kernelSize = 5):
    if min_dist == 0:
        min_dist = image.shape[0]//16
    # Convert to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Apply Gaussian blur to reduce noise
    # img_blur = cv2.GaussianBlur(gray, (kernelSize,kernelSize), 0)

    # Blur the image to reduce noise
    img_blur = cv2.medianBlur(gray, kernelSize)
    # Apply hough transform on the image
    
    
    # gray: Input image (grayscale).
    # circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
    # HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
    # dp = 1: The inverse ratio of resolution.
    # min_dist = gray.rows/16: Minimum distance between detected centers.
    # param_1 = 200: Upper threshold for the internal Canny edge detector.
    # param_2 = 100*: Threshold for center detection.
    # min_radius = 0: Minimum radius to be detected. If unknown, put zero as default.
    # max_radius = 0: Maximum radius to be detected. If unknown, put zero as default.
    # # Apply Hough transform to detect circles
    circles = cv2.HoughCircles(img_blur, method, dp, min_dist,
                            param1, param2, minRadius, maxRadius)
    
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1.3, 30, param1=150, param2=70, minRadius=0, maxRadius=0)

    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, image.shape[0]/64, \
    #     param1=200, param2=10, minRadius=0, maxRadius=30)
    
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    return circles 


def houghTransformLines(image, cannyThresh1 = 50, cannyThresh2 = 200, cannyEdges =None,\
        canny_apertureSize = 3, canny_L2gradient =False,\
        HL_rho = 1, HL_theta = 0, HL_threshold = 150, HL_lines = None, HL_srn = 0, HL_stn = 0):  
    '''
    @brief Finds lines in a binary image using the standard Hough transform.

    The function implements the standard or standard multi-scale Hough transform algorithm for line detection. See <http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm> for a good explanation of Hough transform.
    
    @param cannyEdges	output edge map; single channels 8-bit image, which has the same size as image .
    @param cannyThresh1	first threshold for the hysteresis procedure.
    @param cannyThresh2	second threshold for the hysteresis procedure.
    @param canny_apertureSize	aperture size for the Sobel operator.
    @param canny_L2gradient	a flag, indicating whether a more accurate L_2 norm =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L_1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
    
    @param lines Output vector of lines. Each line is represented by a 2 or 3 element vector \f$(\rho, \theta)\f$ or \f$(\rho, \theta, \textrm{votes})\f$ . \f$\rho\f$ is the distance from the coordinate origin \f$(0,0)\f$ (top-left corner of the image). \f$\theta\f$ is the line rotation angle in radians ( \f$0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\f$ ). \f$\textrm{votes}\f$ is the value of accumulator.
    @param rho Distance resolution of the accumulator in pixels.
    @param theta Angle resolution of the accumulator in radians.
    @param threshold Accumulator threshold parameter. Only those lines are returned that get enough votes ( \f$>\texttt{threshold}\f$ ).
    @param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho . The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these parameters should be positive.
    @param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
    @param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines.
    Must fall between 0 and max_theta.
    @param max_theta For standard and multi-scale Hough transform, maximum angle to check for lines.
    Must fall between min_theta and CV_PI.

    The Hough Line Transform is a transform used to detect straight lines.
    To apply the Transform, first an edge detection pre-processing is desirable.
    
    Parameters
    ------------
    dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    lines: A vector that will store the parameters (r,θ) of the detected lines
    rho : The resolution of the parameter r in pixels. We use 1 pixel.
    theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    threshold: The minimum number of intersections to "*detect*" a line
    srn and stn: Default parameters to zero. Check OpenCV reference for more info.
    '''      

            # Check if image is loaded fine
    if image is None:
        print('Error opening image!')
        return -1
    
    
    dst = cv2.Canny(image, cannyThresh1, cannyThresh2, cannyEdges, canny_apertureSize, canny_L2gradient)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    
    if HL_theta == 0:
        HL_theta = np.pi / 180
    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    lines = cv2.HoughLines(dst, rho= HL_rho, theta= HL_theta ,\
        threshold= HL_threshold, lines= HL_lines, srn=  HL_srn,stn= HL_stn)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
    return cdst, lines

# Probabilistic Line Transform
def houghTransformLinesP(image, cannyThresh1 = 50, cannyThresh2 = 200, cannyEdges =None,\
        canny_apertureSize = 3, canny_L2gradient =False,\
        HL_rho = 1, HL_theta = 0, HL_threshold = 150, HL_lines = None, HL_minLineLength = 50, HL_maxLineGap = 10):  
    '''
    HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> canny, lines
    @brief Finds line segments in a binary image using the probabilistic Hough transform.
    The function implements the probabilistic Hough transform algorithm for line detection, described in @cite Matas00
    See the line detection example below:
    @include snippets/imgproc_HoughLinesP.cpp This is a sample picture the function parameters have been tuned for: image
    And this is the output of the above program in case of the probabilistic Hough transform: image
    
    
    @param cannyEdges	output edge map; single channels 8-bit image, which has the same size as image .
    @param cannyThresh1	first threshold for the hysteresis procedure.
    @param cannyThresh2	second threshold for the hysteresis procedure.
    @param canny_apertureSize	aperture size for the Sobel operator.
    @param canny_L2gradient	a flag, indicating whether a more accurate L_2 norm =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L_1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
    
    @param lines Output vector of lines. Each line is represented by a 4-element vector \f$(x_1, y_1, x_2, y_2)\f$ , where \f$(x_1,y_1)\f$ and \f$(x_2, y_2)\f$ are the ending points of each detected line segment.
    @param rho Distance resolution of the accumulator in pixels.
    @param theta Angle resolution of the accumulator in radians.
    @param threshold Accumulator threshold parameter. Only those lines are returned that get enough votes ( \f$>\texttt{threshold}\f$ ).
    @param minLineLength Minimum line length. Line segments shorter than that are rejected.
    @param maxLineGap Maximum allowed gap between points on the same line to link them.
    @sa LineSegmentDetector

    The Hough Line Transform is a transform used to detect straight lines.
    To apply the Transform, first an edge detection pre-processing is desirable.
    
    Parameters
    ------------
    dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    lines: A vector that will store the parameters (r,θ) of the detected lines
    rho : The resolution of the parameter r in pixels. We use 1 pixel.
    theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    threshold: The minimum number of intersections to "*detect*" a line
    srn and stn: Default parameters to zero. Check OpenCV reference for more info.
    '''      

            # Check if image is loaded fine
    if image is None:
        print('Error opening image!')
        return -1
    

    dst = cv2.Canny(image, cannyThresh1, cannyThresh2, cannyEdges, canny_apertureSize, canny_L2gradient)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    
    
    if HL_theta == 0:
        HL_theta = np.pi / 180
        
    linesP = cv2.HoughLinesP(dst, HL_rho, HL_theta, HL_threshold, HL_lines, HL_minLineLength, HL_maxLineGap)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        
    return cdstP, linesP