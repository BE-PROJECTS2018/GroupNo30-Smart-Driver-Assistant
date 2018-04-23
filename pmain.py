import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import pickle
import io
import os
import glob
from moviepy.video.fx.all import crop
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
from scipy.interpolate import splprep, splev

imageC = 1
xp,yp=0,0
counter = 0
sign_list=[]
dont=0
timer=0
just=0
save_path = "./cropped/"
font = cv2.FONT_HERSHEY_COMPLEX 
color = (0,255,0)
#%%
def img_pipeline(img):

    global window_search
    global left_fit_prev
    global right_fit_prev
    global frame_count
    global curve_radius
    global offset

    
    # load camera matrix and distortion matrix
    camera = pickle.load(open( "camera_matrix.pkl", "rb" ))
    mtx = camera['mtx']
    dist = camera['dist']
    camera_img_size = camera['imagesize']

    #correct lens distortion
    undist = distort_correct(img,mtx,dist,camera_img_size)
    # get binary image
    binary_img = binary_pipeline(undist)
    #perspective transform
    birdseye, inverse_perspective_transform = warp_image(binary_img)

    if window_search:
        #window_search = False
        #window search
        left_fit,right_fit = track_lanes_initialize(birdseye)
        #store values
        left_fit_prev = left_fit
        right_fit_prev = right_fit

    else:
        #load values
        left_fit = left_fit_prev
        right_fit = right_fit_prev
        #search in margin of polynomials
        left_fit,right_fit,leftx,lefty,rightx,righty = track_lanes_update(birdseye, left_fit,right_fit)

    #save values
    left_fit_prev = left_fit
    right_fit_prev = right_fit

    #draw polygon
    processed_frame = lane_fill_poly(birdseye, undist, left_fit, right_fit,inverse_perspective_transform)

    #update ~twice per second
    if frame_count==0 or frame_count%15==0:
        #measure radii
        curve_radius = measure_curve(birdseye,left_fit,right_fit)
        #measure offset
        offset = vehicle_offset(undist, left_fit, right_fit)


    #printing information to frame
    font = cv.FONT_HERSHEY_TRIPLEX
    processed_frame = cv.putText(processed_frame, 'Radius: '+str(curve_radius)+' m', (30, 40), font, 1, (0,255,0), 2)
    processed_frame = cv.putText(processed_frame, 'Offset: '+str(offset)+' m', (30, 80), font, 1, (0,255,0), 2)
    
    processed_frame = cv.resize(processed_frame, (640, 360))
    processed_frame_2 = Traffic(processed_frame)

    frame_count += 1
    return processed_frame_2
#%%
def Traffic(img):
    
############
    global imageC
    global xp,yp
    global counter
    global sign_list
    global dont
    global timer
    global just
    global save_path
    global font
    global color
    
    
    counter+=1
    print(counter)
    #My Code
    
    
    overlay = img.copy()
    #output2 = img.copy()
    output3 = img.copy()
    # img = cv.GaussianBlur(frame,(5,5),0)
    if(timer==0 and len(sign_list)!=0):
        if(just!=1):
            sign_list.pop(0)
            just=1
        else:
            just=0
            timer = 150
    else:
        timer-=1
    if(dont>0):
        dont-=1
        alpha=0.3
        cv.rectangle(overlay, (0, 270), (640, 360),(0, 0, 0), -1)
        cv.addWeighted(overlay, alpha, output3, 1 - alpha,0, output3)
        offset_x=10
        offset_y=280
        for i,sign_name in enumerate(sign_list):
            sign_img = cv.imread(os.path.join("./traffic-signs-data/signs/",str(sign_name) +".jpg"))
            sign_img = cv.resize(sign_img, (90,60))
            #cv.imshow('asd',sign_img)
            #cv.waitKey()
            output3[offset_y:offset_y+60,offset_x*(i+1)+(90*i):offset_x*(i+1)+(90*(i+1))] = sign_img
            cv.putText(output3, signs[sign_name],(offset_x*(i+1)+(90*i), offset_y+70), font, 0.3,(255,255,255),1,cv.LINE_AA)
            # Display frame
        ################cv.imshow(window_name,output3)
        #Write method from VideoWriter. This writes frame to video file
        return output3
        #Read next frame from video
        '''
        state, img = video_read.read()
        #Check if any key is pressed.
        k = cv.waitKey(display_rate)
        #Check if ESC key is pressed. ASCII Keycode of ESC=27
        if k == esc_keycode:
            #Destroy Window
            cv.destroyWindow(window_name)
            break
        continue
        '''
        # MSER
        """""
    mser = cv.MSER_create()
    regions = mser.detectRegions(img)
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv.polylines(vis, hulls, 1, (0, 255, 0))
    cv.imshow('img', vis)
    """

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # cv.imshow('hsv', hsv)

    # range
    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv.inRange(hsv, lower_red_2, upper_red_2)

    mask = mask1 + mask2

    red = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow('res1', red)

    gray = cv.cvtColor(red, cv.COLOR_BGR2GRAY)
    # cv.imshow('res2', gray)

    # blur = cv.GaussianBlur(gray,(3,3),0)
    # cv.imshow('res3', blur)
    blur = cv.GaussianBlur(gray, (3, 3), 0)

    kernel = np.ones((3, 3), np.uint8)

    close = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)
    # cv.imshow('res3', close)
    median = cv.medianBlur(close, 3)

    # cv.imshow("i",np.hstack([gray,median,close]))
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharp = cv.filter2D(median, -1, kernel)
    # cv.imshow('res5', sharp)

    im2, contours, hierarchy = cv.findContours(close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        area = cv.contourArea(contour)
        if ((len(approx) > 8) & (area > 130)):
            contour_list.append(contour)

    # smoothening of contours
    '''
    smoothened = []
    for contour in contour_list:
        x_img, y_img = contour.T
        # Convert from numpy arrays to normal arrays
        x_img = x_img.tolist()[0]
        y_img = y_img.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x_img, y_img], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))


    cv.drawContours(output, smoothened, -1, (0, 255, 0), 3)
    cv.fillPoly(output, pts=smoothened, color=(0, 255, 0))
    '''
    # Getting circles

    threshold = 0.50
    circle_list = []
    for contour in contour_list:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        metric = 4 * 3.14 * area / pow(perimeter, 2)
        if (metric > threshold):
            circle_list.append(contour)
    #print(circle_list)
    # cv.drawContours(output2, circle_list, -1, (0, 255, 0), 3)
    # cv.fillPoly(output2, pts=circle_list, color=(0, 255, 0))

    # Drawing Rectangle
    '''
    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(circle_list, hierarchy):
        (x_img, y_img, w_img, h_img) = cv.boundingRect(contour)
        min_x, max_x = min(x_img, min_x), max(x_img + w_img, max_x)
        min_y, max_y = min(y_img, min_y), max(y_img + h_img, max_y)
        if w_img > 80 and h_img > 80:
            cv.rectangle(img, (x_img, y_img), (x_img + w_img, y_img + h_img), (255, 0, 0), 1)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)
    '''
    
    #Overlay
    alpha=0.3
    cv.rectangle(overlay, (0, 270), (640, 360),(0, 0, 0), -1)
    cv.addWeighted(overlay, alpha, output3, 1 - alpha,0, output3)
    offset_x=10
    offset_y=280
    xm, ym, wm, hm, = 0, 0, 0, 0,
    for contour in circle_list:
        # get rectangle bounding contour
        [x_img, y_img, w_img, h_img] = cv.boundingRect(contour)
        #print("m = ", [xm, ym, wm, hm])
        #print([x_img, y_img, w_img, h_img])
        if(xp==x_img and yp==y_img):
            continue
        elif (xm == 0):
            xm, ym, wm, hm = x_img, y_img, w_img, h_img
        else:
            if ((abs(xm - x_img) < 30) and (abs(ym - y_img) < 30) and (abs(wm - w_img)>3) and (abs(hm -h_img)>3)):
                # xmean, ymean = (xm+x_img)//2,(ym+y_img)//2
                # wmax, hmax = max(wm,w_img),max(hm,h_img)
                if (wm > w_img):
                    car=[]
                    crop = img[(ym):(ym + hm), (xm):(xm + wm)].copy()
                    xp,yp = xm,ym
                else:
                    car=[]
                    crop = img[(y_img):(y_img + h_img), (x_img):(x_img + w_img)].copy()
                    xp, yp = x_img, y_img
                cimg = cv.resize(crop, (32,32))
                #cimg = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)
                car.append(cimg)
                new_image_processed = preprocess(np.asarray(car))
                #print(crop.shape)
                #print(cimg.shape)
                #print(new_image_processed.shape)
                y_prob, y_pred = y_predict_model1(new_image_processed)
                if(y_prob[0][0]<0.999):
                    continue
                #print(len(new_image_processed))
                dont = 50
                if y_pred[0][0] not in sign_list:
                    timer = 300
                    sign_list.append(y_pred[0][0])
                    print(sign_list)
                    print("dont=",str(dont))
                    print (signs[y_pred[0][0]])
                    print(y_pred[0][0])
                    print (y_prob[0][0])
                #cv.imwrite(os.path.join(save_path, str(imageC) + ".jpg"), crop)
                #print("IMAGE SAVED FOR CLASSIFICATION ", str(imageC))
                cv.rectangle(output3, (x_img, y_img), (x_img + w_img, y_img + h_img), (0, 255, 0), 1)
        #cv.putText(output3,(x_img - w_img, y_img - h_img),font,255,1.0,color)
                cv.putText(output3, signs[y_pred[0][0]],(x_img - w_img, y_img - h_img), font, 0.2,(0,255,0),1,cv.LINE_AA)
                imageC += 1
        xm, ym, wm, hm = x_img, y_img, w_img, h_img
        # draw rectangle around contour on original image
    for i,sign_name in enumerate(sign_list):
        sign_img = cv.imread(os.path.join("./traffic-signs-data/signs/",str(sign_name) +".jpg"))
        sign_img = cv.resize(sign_img, (90,60))
        #cv.imshow('asd',sign_img)
        #cv.waitKey()
        output3[offset_y:offset_y+60,offset_x*(i+1)+(90*i):offset_x*(i+1)+(90*(i+1))] = sign_img
        cv.putText(output3, signs[sign_name],(offset_x*(i+1)+(90*i), offset_y+70), font, 0.3,(255,255,255),1,cv.LINE_AA)
            
            #cv.putText(output3, str(y_prob[0][0]),(x_img - w_img + 50, y_img - h_img), font, 0.2,(0,255,0),1,cv.LINE_AA)

        # Write frame
        # if((length/counter)%100 == 0):
        #print(counter)

        # Display frame
    return output3
#%%

def distort_correct(img,mtx,dist,camera_img_size):
    img_size1 = (img.shape[1],img.shape[0])
    #print(img_size1)
    #print(camera_img_size)
    assert (img_size1 == camera_img_size),'image size is not compatible'
    undist = cv.undistort(img, mtx, dist, None, mtx)
    return undist


def binary_pipeline(img):

    img_copy = cv.GaussianBlur(img, (3, 3), 0)
    #img_copy = np.copy(img)

    # color channels
    s_binary = hls_select(img_copy, sthresh=(140, 255), lthresh=(120, 255))
    #red_binary = red_select(img_copy, thresh=(200,255))

    # Sobel x
    x_binary = abs_sobel_thresh(img_copy,thresh=(25, 200))
    y_binary = abs_sobel_thresh(img_copy,thresh=(25, 200), orient='y')
    xy = cv.bitwise_and(x_binary, y_binary)

    #magnitude & direction
    mag_binary = mag_threshold(img_copy, sobel_kernel=3, thresh=(30,100))
    dir_binary = dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))

    # Stack each channel
    gradient = np.zeros_like(s_binary)
    gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    final_binary = cv.bitwise_or(s_binary, gradient)

    return final_binary


def hls_select(img, sthresh=(0, 255),lthresh=()):
    # 1) Convert to HLS color space
    hls_img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                 & (L > lthresh[0]) & (L <= lthresh[1])] = 1
    return binary_output


def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the xy magnitude
    mag = np.sqrt(x**2 + y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale = np.max(mag)/255
    eightbit = (mag/scale).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(eightbit)
    binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] =1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel))
    y = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(y, x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return binary_output


def warp_image(img):

    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]

    #the "order" of points in the polygon you are defining does not matter
    #but they need to match the corresponding points in destination_points!
    source_points = np.float32([
    [0.117 * x, y],
    [(0.5 * x) - (x*0.078), (2/3)*y],
    [(0.5 * x) + (x*0.078), (2/3)*y],
    [x - (0.117 * x), y]
    ])

#     #chicago footage
#     source_points = np.float32([
#                 [300, 720],
#                 [500, 600],
#                 [700, 600],
#                 [850, 720]
#                 ])

#     destination_points = np.float32([
#                 [200, 720],
#                 [200, 200],
#                 [1000, 200],
#                 [1000, 720]
#                 ])

    destination_points = np.float32([
    [0.25 * x, y],
    [0.25 * x, 0],
    [x - (0.25 * x), 0],
    [x - (0.25 * x), y]
    ])

    perspective_transform = cv.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv.getPerspectiveTransform( destination_points, source_points)

    warped_img = cv.warpPerspective(img, perspective_transform, image_size, flags=cv.INTER_LINEAR)

    #print(source_points)
    #print(destination_points)

    return warped_img, inverse_perspective_transform



def track_lanes_initialize(binary_warped):

    global window_search

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # we need max for each half of the histogram. the example above shows how
    # things could be complicated if didn't split the image in half
    # before taking the top 2 maxes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    # this will throw an error in the height if it doesn't evenly divide the img height
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []


    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3)
        cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit,right_fit



def track_lanes_update(binary_warped, left_fit,right_fit):

    global window_search
    global frame_count

    # repeat window search to maintain stability
    if frame_count % 10 == 0:
        window_search=True


    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    return left_fit,right_fit,leftx,lefty,rightx,righty


def lane_fill_poly(binary_warped,undist,left_fit,right_fit,inverse_perspective_transform):

    # Generate x and y values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = get_val(ploty,left_fit)
    right_fitx = get_val(ploty,right_fit)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast x and y for cv.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp using inverse perspective transform
    newwarp = cv.warpPerspective(color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0]))
    # overlay
    #newwarp = cv.cvtColor(newwarp, cv.COLOR_BGR2RGB)
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

# A function to get quadratic polynomial output
def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]


def measure_curve(binary_warped,left_fit,right_fit):

    # generate y values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # measure radius at the maximum y value, or bottom of the image
    # this is closest to the car
    y_eval = np.max(ploty)

    # coversion rates for pixels to metric
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # x positions lanes
    leftx = get_val(ploty,left_fit)
    rightx = get_val(ploty,right_fit)

    # fit polynomials in metric
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # calculate radii in metric from radius of curvature formula
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # averaged radius of curvature of left and right in real world space
    # should represent approximately the center of the road
    curve_rad = round((left_curverad + right_curverad)/2)

    return curve_rad


def vehicle_offset(img,left_fit,right_fit):

    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    xm_per_pix = 3.7/700
    image_center = img.shape[1]/2

    ## find where lines hit the bottom of the image, closest to the car
    left_low = get_val(img.shape[0],left_fit)
    right_low = get_val(img.shape[0],right_fit)

    # pixel coordinate for center of lane
    lane_center = (left_low+right_low)/2.0

    ## vehicle offset
    distance = image_center - lane_center

    ## convert to metric
    return (round(distance*xm_per_pix,5))


#%%
def y_predict_model1(Input_data, top_k=5):
    """
    Generates the predictions of the model over the input data, and outputs the top softmax probabilities.
        Parameters:
            X_data: Input data.
            top_k (Default = 5): The number of top softmax probabilities to be generated.
    """
    num_examples = len(Input_data)
    y_pred = np.zeros((num_examples, top_k), dtype=np.int32)
    y_prob = np.zeros((num_examples, top_k))
        #VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
    y_prob, y_pred = sess.run(tf.nn.top_k(tf.nn.softmax(VGGNet_Model.logits), k=top_k), 
                             feed_dict={x:Input_data, keep_prob:1, keep_prob_conv:1})
    return y_prob, y_pred

##y_prob, y_pred = y_predict_model(new_test_images_preprocessed)


global window_search
global frame_count
window_search = True
frame_count = 0
#%%
#chicago footage

for filename in ['mini.mp4']:
    with tf.Session() as sess:
        VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
        clip = VideoFileClip(filename)#.subclip((3,25),(3,35))
        #clip_crop = crop(clip, x1=320, y1=0, x2=1600, y2=720)
        out= clip.fl_image(img_pipeline)
        #out = clip_crop.fl_image(img_pipeline)
        out.write_videofile('p1_'+filename, audio=False, verbose=False)
    print('Success!')
#%%    
    '''
clip.reader.close()
clip.audio.reader.close_proc()

target = "camera_matrix.pkl"
if os.path.getsize(target) > 0:
    cammat = pickle.load(open( "./camera_matrix.pkl", "rb" ))
    print(camera)
else:
    print("File is empty")


cammat.close()
'''