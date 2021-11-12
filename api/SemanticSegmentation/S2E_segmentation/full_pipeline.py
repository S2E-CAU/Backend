#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings(action='ignore')

import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
#import albumentations as A

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:

'''
def model_and_predict(image):
    # Backbone & preprocess
    BACKBONE = 'seresnet152'
    #preprocess_input = sm.get_preprocessing(BACKBONE)
    
    # build model
    loss = sm.losses.bce_jaccard_loss
    metrics = sm.metrics.FScore()
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile('Adam', loss=loss, metrics=[metrics])
    model.load_weights('seresnet152_gen_roof_all.h5')

    # read image
    test_img = cv2.imread('dongjak.jpg', cv2.IMREAD_COLOR)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    #plt.imshow(test_img, cmap='gray')
    test_img = np.expand_dims(test_img, axis=0)

    # predict image
    prediction = model.predict(test_img)
    prediction_image = prediction.reshape(512,512,1)
    prediction_image = prediction_image > 0.5
    prediction_image = prediction_image * 255.
    #plt.imshow(prediction_image, cmap='gray')
    return prediction_image

'''
# In[13]:


image_pth = 'dongjak.jpg'
mask_pth = 'dongjak_mask.jpg'


# In[14]:


def mouse_crop(img, mask, num):
    # get image and mask
    # use mouse to crop ROI
    # save the cropped ROI for image and mask
    
    
    img2 = img.copy()
    # variables
    ix = -1
    iy = -1
    drawing = False
    #
    if num == 1:
        x1=161
        x2=231
        y1=239
        y2=327

    if num ==2:
        x1 = 203
        x2 = 243
        y1 = 139
        y2 = 192

    if num==3:
        x1 = 202
        x2 = 258
        y1 = 70
        y2 = 127
    
    img_crop = img2[y1-1:y2+1, x1-1:x2+1]
   
    mask_crop = mask[y1-1:y2+1, x1-1:x2+1]

    return img_crop, mask_crop



# In[7]:


def mask_clean(mask_crop):
    # flood
    img = mask_crop
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), dtype=np.uint8)
    holes = cv2.floodFill(thresh.copy(), mask, (0, 0), 255)[1]
    holes = ~holes
    thresh[holes == 255] = 255

    # close opening
    img = cv2.erode(thresh,(3,3),iterations=3)
    img = cv2.dilate(img,(3,3),iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    (thresh, binRed) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    #plt.figure(figsize=(8,8))
    #plt.imshow(opening, cmap='gray')
    return mask_crop




# In[8]:


def crop_one_building(img_crop, mask_crop):
    # from the ROI we crop the largest contour assuming it to be the building we want to predict
    
    image = img_crop  
    mask = mask_crop
    mask2= mask.copy()
    
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Apply cv2.threshold() to get a binary image
    ret, thresh = cv2.threshold(gray_image, 100, 255, 0)
    
    # Find contours:
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = []
    #for c in contours:
    #    x,y,w,h = cv2.boundingRect(c)
    #    if not (w < 10 or h < 10): 
    #        cnts.append(c)
    #print(contours)
    
    max_value = max(contours, key = len)
    max_index = contours.index(max_value)
    #print(max_index)
    cv2.drawContours(mask2, contours, max_index, (100, 255, 255), -1)

    x, y = [], []
    for contour in contours[max_index]:
        x.append(contour[0][0])
        y.append(contour[0][1])

    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
    img_cropped = image[y1-2:y2+2, x1-2:x2+2]
    mask_cropped= mask[y1-2:y2+2, x1-2:x2+2]

    return img_cropped, mask_cropped


# In[9]:


def DP_method(mask_cropped):
    # use DP method onto the mask
    # maybe find corners first and if they are below a certain number use DP
    # because it is useful for rectanglur images rather than polygons ?
    # https://stackoverflow.com/questions/50984205/how-to-find-corners-points-of-a-shape-in-an-image-in-opencv
    # DP METHOD
    mask = mask_cropped
    #img = img_cropped
    mask_zero = np.zeros(mask.shape, np.uint8)

    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply cv2.threshold() to get a binary image
    ret, thresh = cv2.threshold(gray_image, 100, 255, 0)

    # Find contours:
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if not (w < 10 or h < 10): 
            cnts.append(c)

    for i in range(len(cnts)):
        epsilon = 0.02*cv2.arcLength(cnts[i], True)
        approx = cv2.approxPolyDP(cnts[i], epsilon, True)
        vtc = len(approx)
        print(vtc)

        if vtc <= 8:
            cv2.drawContours(mask_zero, [approx], 0, (255, 255, 255), -1)
        else:
            cv2.drawContours(mask_zero, cnts, i, (255, 255, 255), -1)

    return mask_zero

def solar_panel(img_cropped, mask_cropped):
    # draw rectangles on building and visualize
    # from the number of panels get the ammount of potential energy
    mask = mask_cropped
    mask2 = mask.copy()
    ret, thresh = cv2.threshold(mask2, 100, 255, 0)
    mask2 = thresh

    number = 0
    j = 0

    while(j < mask.shape[0]): # y
        i =0
        start = (0, j)  # x y
        end = (10, j+9)  # x y

        if start[1]!=0:
            start = (start[0], start[1]+1)
            end = (end[0], end[1]+1)

        while(i < mask.shape[1]): # x

            if (mask2[start[1]:end[1], start[0]:end[0]] != 0).all():
                cv2.rectangle(mask2, start, end, (100,100,100), 1)
                cv2.rectangle(mask2, (start[0]+1, start[1]+1),(end[0]-1, end[1]-1), (255,255,200), -1)

                w = start[0]
                h = start[1]


                while h < end[1]:
                    cv2.rectangle(mask2, (w, h), (w + 5, h + 3), (100, 100, 100), 1)
                    cv2.rectangle(mask2, (w + 5, h), (w + 10, h + 3), (100, 100, 100), 1)
                    h += 3


                start = (start[0]+18, start[1])
                end = (end[0]+18, end[1])
                i = end[0]

                number += 1

            else:
                start = (start[0]+1, start[1])
                end = (end[0]+1, end[1])
                i = end[0]

        j = end[1]

    print(number)
    
    return mask2, number

    
    # 1. the area of the building using countNonZero
    # 2. number of panels x Pn x Pv = solar potential
    # 3. (think)
    
    #ret,thresh=cv2.threshold(img,133,255,cv2.THRESH_BINARY_INV)
    
    #img = cv2.imread("Crop_out_white.jpg",0)
    #ret, thresh = cv2.threshold(img, 100, 255, 0)
    #pixel=cv2.countNonZero(thresh)
    #print(img.shape[0]*img.shape[1])
    
    #pixel
    #one_pixel = 173.312/1433
    #one_pixel
    #area = pixel*10 #m^2
    #solar_potential=Panel*Pn*PV #Pn=kw,태양광 패널의 공칭 용량 / PV=kWh,kWp 출력


def execute(img, mask,num):
    
    img_crop, mask_crop = mouse_crop(img, mask, num)
    mask_crop = mask_clean(mask_crop)
    img_cropped, mask_cropped = crop_one_building(img_crop, mask_crop)
    mask_cropped = DP_method(mask_cropped)
    #visualize(img_cropped, mask_cropped)
    mask, number = solar_panel(img_cropped, mask_cropped)
    
    return mask, number

