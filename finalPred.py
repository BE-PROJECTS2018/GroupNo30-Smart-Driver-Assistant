# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 01:16:53 2018

@author: Aditya
"""
'''
new_test_images = []
path = './traffic-signs-data/new_test_images/'
for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_test_images.append(img)
    
#%%
new_test_images_preprocessed = preprocess(np.asarray(new_test_images))

#%%
y_prob, y_pred = y_predict_model(new_test_images_preprocessed)

#%%
for i in range(len(new_test_images_preprocessed)):
    print (signs[y_pred[i][0]])
    
#%%
def preprocess1(data):
    """
    Applying the preprocessing steps to the input data.
        Parameters:
            data: An np.array compatible with plt.imshow.
    """
    #gray_images = list(map(gray_scale, data))
    equalized_images = list(map(local_histo_equalize, data))
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = image_normalize(img)
    normalized_images = normalized_images[..., None]
    print(normalized_images.shape)
    return normalized_images

#%%
print(new_test_images_preprocessed.shape)
'''
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
'''
y_prob, y_pred = y_predict_model(new_test_images_preprocessed)
'''
#%%
save_path = "./cropped/"
#Path of video file to be read
video_read_path='chava_project_video.mp4'
font = cv2.FONT_HERSHEY_COMPLEX 
color = (0,255,0)
#Path of video file to be written
video_write_path='chava_project_video_processed.avi'

#Window Name
window_name='Input Video'

#Escape ASCII Keycode
esc_keycode=27

#Create an object of VideoCapture class to read video file
video_read = cv2.VideoCapture(video_read_path)
    # Check if video file is loaded successfully
if (video_read.isOpened()== True):
    #Frames per second in videofile. get method in VideoCapture class.
    fps = video_read.get(cv2.CAP_PROP_FPS)
    #Width and height of frames in video file
    size = (int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #Create an object of VideoWriter class to write video file.
    #cv2.CV_FOURCC('I','4','2','0') = uncompressed YUV, 4:2:0 chroma subsampled. (.avi)
    #cv2.CV_FOURCC('P','I','M','1') = MPEG-1(.avi)
    #cv2.CV_FOURCC('M','J','P','G') = motion-JPEG(.avi)
    #cv2.CV_FOURCC('T','h_img','E','O') = Ogg-Vorbis(.ogv)
    #cv2.CV_FOURCC('F','L','V','1') = Flash video (.flv)
    #cv2.CV_FOURCC('M','P','4','V') = MPEG encoding (.mp4)
    #Also this form is too valid cv2.VideoWriter_fourcc(*'MJPG')
    #video_write = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    video_write = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    #Set display frame rate
    display_rate = (int) (1/fps * 1000)
    #Create a Window
    #cv2.WINDOW_NORMAL = Enables window to resize.
    #cv2.WINDOW_AUTOSIZE = Default flag. Auto resizes window size to fit an image.
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    counter = 0
    #Read first frame from video. Return Boolean value if it succesfully reads the frame in state and captured frame in cap_frame
    state, img = video_read.read()
    #Loop untill all frames from video file are read
    imageC = 1
    xp,yp=0,0
    
    sign_list=[]
    dont=0
    timer=0
    just=0
    with tf.Session() as sess:
        VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
        while state:
            counter+=1
            print(counter)
            #My Code
            
            #img = cv.resize(img, ( 720,1280))
            overlay = img.copy()
            #output2 = img.copy()
            output3 = img.copy()
            # img = cv2.GaussianBlur(frame,(5,5),0)
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
                cv2.rectangle(overlay, (0, 270), (640, 360),(0, 0, 0), -1)
                cv2.addWeighted(overlay, alpha, output3, 1 - alpha,0, output3)
                offset_x=10
                offset_y=280
                for i,sign_name in enumerate(sign_list):
                    sign_img = cv2.imread(os.path.join("./traffic-signs-data/signs/",str(sign_name) +".jpg"))
                    sign_img = cv2.resize(sign_img, (90,60))
                    #cv2.imshow('asd',sign_img)
                    #cv2.waitKey()
                    output3[offset_y:offset_y+60,offset_x*(i+1)+(90*i):offset_x*(i+1)+(90*(i+1))] = sign_img
                    cv2.putText(output3, signs[sign_name],(offset_x*(i+1)+(90*i), offset_y+70), font, 0.3,(255,255,255),1,cv2.LINE_AA)
                    # Display frame
                cv2.imshow(window_name,output3)
                #Write method from VideoWriter. This writes frame to video file
                video_write.write(output3)
                #Read next frame from video
                state, img = video_read.read()
                #Check if any key is pressed.
                k = cv2.waitKey(display_rate)
                #Check if ESC key is pressed. ASCII Keycode of ESC=27
                if k == esc_keycode:
                    #Destroy Window
                    cv2.destroyWindow(window_name)
                    break
                continue
            # MSER
            """""
            mser = cv2.MSER_create()
            regions = mser.detectRegions(img)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            cv2.polylines(vis, hulls, 1, (0, 255, 0))
            cv2.imshow('img', vis)
            """
    
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # cv2.imshow('hsv', hsv)
    
            # range
            lower_red_1 = np.array([0, 50, 50])
            upper_red_1 = np.array([10, 255, 255])
            lower_red_2 = np.array([170, 50, 50])
            upper_red_2 = np.array([180, 255, 255])
    
            mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
            mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
            mask = mask1 + mask2
    
            red = cv2.bitwise_and(img, img, mask=mask)
            # cv2.imshow('res1', red)
    
            gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('res2', gray)
    
            # blur = cv2.GaussianBlur(gray,(3,3),0)
            # cv2.imshow('res3', blur)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
            kernel = np.ones((3, 3), np.uint8)
    
            close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow('res3', close)
            median = cv2.medianBlur(close, 3)
    
            # cv2.imshow("i",np.hstack([gray,median,close]))
            # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # sharp = cv2.filter2D(median, -1, kernel)
            # cv2.imshow('res5', sharp)
    
            im2, contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
            contour_list = []
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                area = cv2.contourArea(contour)
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
        
        
            cv2.drawContours(output, smoothened, -1, (0, 255, 0), 3)
            cv2.fillPoly(output, pts=smoothened, color=(0, 255, 0))
            '''
            # Getting circles
    
            threshold = 0.50
            circle_list = []
            for contour in contour_list:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                metric = 4 * 3.14 * area / pow(perimeter, 2)
                if (metric > threshold):
                    circle_list.append(contour)
            #print(circle_list)
            # cv2.drawContours(output2, circle_list, -1, (0, 255, 0), 3)
            # cv2.fillPoly(output2, pts=circle_list, color=(0, 255, 0))
    
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
                (x_img, y_img, w_img, h_img) = cv2.boundingRect(contour)
                min_x, max_x = min(x_img, min_x), max(x_img + w_img, max_x)
                min_y, max_y = min(y_img, min_y), max(y_img + h_img, max_y)
                if w_img > 80 and h_img > 80:
                    cv2.rectangle(img, (x_img, y_img), (x_img + w_img, y_img + h_img), (255, 0, 0), 1)
        
            if max_x - min_x > 0 and max_y - min_y > 0:
                cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)
            '''
            
            #Overlay
            alpha=0.3
            cv2.rectangle(overlay, (0, 270), (640, 360),(0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, output3, 1 - alpha,0, output3)
            offset_x=10
            offset_y=280
            xm, ym, wm, hm, = 0, 0, 0, 0,
            for contour in circle_list:
                # get rectangle bounding contour
                [x_img, y_img, w_img, h_img] = cv2.boundingRect(contour)
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
                        cimg = cv2.resize(crop, (32,32))
                        #cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2GRAY)
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
                        #cv2.imwrite(os.path.join(save_path, str(imageC) + ".jpg"), crop)
                        #print("IMAGE SAVED FOR CLASSIFICATION ", str(imageC))
                        cv2.rectangle(output3, (x_img, y_img), (x_img + w_img, y_img + h_img), (0, 255, 0), 1)
                #cv2.putText(output3,(x_img - w_img, y_img - h_img),font,255,1.0,color)
                        cv2.putText(output3, signs[y_pred[0][0]],(x_img - w_img, y_img - h_img), font, 0.2,(0,255,0),1,cv2.LINE_AA)
                        imageC += 1
                xm, ym, wm, hm = x_img, y_img, w_img, h_img
                # draw rectangle around contour on original image
            for i,sign_name in enumerate(sign_list):
                sign_img = cv2.imread(os.path.join("./traffic-signs-data/signs/",str(sign_name) +".jpg"))
                sign_img = cv2.resize(sign_img, (90,60))
                #cv2.imshow('asd',sign_img)
                #cv2.waitKey()
                output3[offset_y:offset_y+60,offset_x*(i+1)+(90*i):offset_x*(i+1)+(90*(i+1))] = sign_img
                cv2.putText(output3, signs[sign_name],(offset_x*(i+1)+(90*i), offset_y+70), font, 0.3,(255,255,255),1,cv2.LINE_AA)
                
                #cv2.putText(output3, str(y_prob[0][0]),(x_img - w_img + 50, y_img - h_img), font, 0.2,(0,255,0),1,cv2.LINE_AA)
    
            # Write frame
            # if((length/counter)%100 == 0):
            #print(counter)
    
            # Display frame
            cv2.imshow(window_name,output3)
            #Write method from VideoWriter. This writes frame to video file
            video_write.write(output3)
            #Read next frame from video
            state, img = video_read.read()
            #Check if any key is pressed.
            k = cv2.waitKey(display_rate)
            #Check if ESC key is pressed. ASCII Keycode of ESC=27
            if k == esc_keycode:
                #Destroy Window
                cv2.destroyWindow(window_name)
                break
    #Closes Video file
    video_read.release()
    video_write.release()
else:
    print("Error opening video stream or file")