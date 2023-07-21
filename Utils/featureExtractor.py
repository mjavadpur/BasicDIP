import cv2


def extract_hog(image, template):
    
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    hog = cv2.HOGDescriptor()
    features_image = hog.compute(image)
    features_templ = hog.compute(template)
    return features_image, features_templ

def extract_sift(image, template):
    
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints_image, descriptors_image = sift.detectAndCompute(image, None)
    keypoints_templ, descriptors_templ = sift.detectAndCompute(template, None)        
    return keypoints_image, descriptors_image, keypoints_templ, descriptors_templ

def extract_surf(image, template):
    
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    surf = cv2.SURF_create()
    keypoints_image, descriptors_image = surf.detectAndCompute(image, None)
    keypoints_templ, descriptors_templ = surf.detectAndCompute(template, None)
    return keypoints_image, descriptors_image, keypoints_templ, descriptors_templ

def extract_orb(image, template):
    
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    keypoints_image, descriptors_image = orb.detectAndCompute(image, None)
    keypoints_templ, descriptors_templ = orb.detectAndCompute(template, None)
    return keypoints_image, descriptors_image, keypoints_templ, descriptors_templ

def extract_lbp(image, template):
    
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    radius = 1
    n_points = 8 * radius
    lbp = cv2.ximgproc.createLBPHFaceRecognizer(radius, n_points)
    _, features_image = lbp.compute(image)
    _, features_templ = lbp.compute(template)
    return features_image, features_templ

def match_template(image, template, method):
    
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
    # Create a brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2) # for sift, surf, orb

    # Extract features from the image
    match method:
        case 'hog':
            f_i, f_t = extract_hog(image, template)
            # Perform template matching using HOG features                
            result = cv2.matchTemplate(f_i, f_t, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
            # Draw the best match on the image
            image_with_match = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            template_height, template_width = template.shape[:2]
            cv2.rectangle(image_with_match, max_loc, (max_loc[0] + template_width, max_loc[1] + template_height), (0, 0, 255), 2)
    
            return image_with_match
        case 'sift':
            k_i, d_i, k_t, d_t = extract_sift(image, template)
            # Match the descriptors
            matches = matcher.match(d_i, d_t)
    
            # Sort the matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
    
            # Draw the best match on the image
            best_match = matches[0]
            image_with_match = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(image_with_match, (int(k_i[best_match.queryIdx].pt[0]), int(k_i[best_match.queryIdx].pt[1])), (int(k_t[best_match.trainIdx].pt[0]), int(k_t[best_match.trainIdx].pt[1])), (0, 0, 255), 2)
            
            return image_with_match
            
        case 'surf':
            k_i, d_i, k_t, d_t = extract_surf(image, template)
            matches = matcher.match(d_i, d_t)
    
            # Sort the matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
    
            # Draw the best match on the image
            best_match = matches[0]
            image_with_match = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(image_with_match, (int(k_i[best_match.queryIdx].pt[0]), int(k_i[best_match.queryIdx].pt[1])), (int(k_t[best_match.trainIdx].pt[0]), int(k_t[best_match.trainIdx].pt[1])), (0, 0, 255), 2)
            
            return image_with_match
                            
        case 'orb':
            k_i, d_i, k_t, d_t = extract_orb(image, template)
            matches = matcher.match(d_i, d_t)
    
            # Sort the matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
    
            # Draw the best match on the image
            best_match = matches[0]
            image_with_match = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(image_with_match, (int(k_i[best_match.queryIdx].pt[0]), int(k_i[best_match.queryIdx].pt[1])), (int(k_t[best_match.trainIdx].pt[0]), int(k_t[best_match.trainIdx].pt[1])), (0, 0, 255), 2)
            
            return image_with_match
                            
            
        case 'lbp':
            f_i, f_t = extract_lbp(image, template)
            # Perform template matching using HOG features                
            result = cv2.matchTemplate(f_i, f_t, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
            # Draw the best match on the image
            image_with_match = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            template_height, template_width = template.shape[:2]
            cv2.rectangle(image_with_match, max_loc, (max_loc[0] + template_width, max_loc[1] + template_height), (0, 0, 255), 2)
    
            return image_with_match                                
        