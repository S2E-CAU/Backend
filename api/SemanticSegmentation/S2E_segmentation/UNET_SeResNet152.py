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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import albumentations as A
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[3]:


#get_ipython().system('nvidia-smi')
#!kill 105780


# In[4]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[5]:


BACKBONE = 'seresnet152'
preprocess_input = sm.get_preprocessing(BACKBONE)


# In[6]:


# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     #shear_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     brightness_range=[0.8, 1.2],
                     zoom_range=0.5,
                     fill_mode='constant',
                     validation_split=0.2)

image_datagen = ImageDataGenerator( **data_gen_args)
mask_datagen = ImageDataGenerator(rescale=1./255, **data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
batch_size=16
train_path='roof_all'
image_folder='image'
mask_folder='labels'
image_color_mode='rgb'
mask_color_mode='grayscale'
target_size=(512,512)
seed=1
#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)
image_train_generator = image_datagen.flow_from_directory(
    train_path,
    classes = [image_folder],
    class_mode = None,
    color_mode = image_color_mode,
    target_size = target_size,
    subset='training',
    batch_size = batch_size,
    seed = seed)

image_val_generator = image_datagen.flow_from_directory(
    train_path,
    classes = [image_folder],
    class_mode = None,
    color_mode = image_color_mode,
    target_size = target_size,
    subset='validation',
    batch_size = batch_size,
    seed = seed)

mask_train_generator = mask_datagen.flow_from_directory(
    train_path,
    classes = [mask_folder],
    class_mode = None,
    color_mode = mask_color_mode,
    target_size = target_size,
    subset='training',
    batch_size = batch_size,
    seed = seed)

mask_val_generator = mask_datagen.flow_from_directory(
    train_path,
    classes = [mask_folder],
    class_mode = None,
    color_mode = mask_color_mode,
    target_size = target_size,
    subset='validation',
    batch_size = batch_size,
    seed = seed)

train_gen = zip(image_train_generator, mask_train_generator)
val_gen = zip(image_val_generator, mask_val_generator)


# In[6]:


#loss = sm.losses.binary_focal_dice_loss
#loss = sm.losses.bce_dice_loss
loss = sm.losses.bce_jaccard_loss #binary_focal_jaccard_loss
#loss = sm.losses.DiceLoss()
#metrics = sm.metrics.iou_score
metrics = sm.metrics.FScore()
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=loss, metrics=[metrics])
model.summary()


# In[8]:


callbacks = [
    #keras.callbacks.ModelCheckpoint('./seresnet152_gen_roof_all.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)
]

training_samples = image_train_generator.n
validation_samples = image_val_generator.n

history = model.fit(
    train_gen,
    steps_per_epoch=training_samples // batch_size,
    epochs=200,
    validation_data=val_gen, 
    validation_steps=validation_samples // batch_size,
    #workers=1,
    shuffle=True,
    callbacks=callbacks
)


# In[9]:


#model.save_weights("seresnet152_gen_roof_all2.h5")


# In[10]:


# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['f1-score'])
plt.plot(history.history['val_f1-score'])
plt.title('Model iou')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[7]:


model.load_weights('seresnet152_gen_roof_all.h5')


# In[19]:


#Test on a different image
#READ EXTERNAL IMAGE...
#test_img = cv2.imread('C:\\Users\\User\\Desktop\\Jupyter\\rooftop_dataset\\mini_data\\LC_AP_37607046_078.tif', cv2.IMREAD_COLOR)       
#test_img = cv2.imread('LC_AP_37705083_001.tif', cv2.IMREAD_COLOR)
test_img = cv2.imread('dongjak.jpg', cv2.IMREAD_COLOR)

test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
#test_img = np.reshape(test_img,(512,512,1))
plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)


# In[20]:


prediction_image1 = prediction.reshape(512,512,1)
plt.imshow(prediction_image1, cmap='gray')


# In[21]:


prediction_image1 = prediction.reshape(512,512,1)
prediction_image1 = prediction_image1 > 0.5
prediction_image1 = prediction_image1 * 255.
plt.imshow(prediction_image1, cmap='gray')

#cv2.imwrite('.tif', grayImg)


# In[22]:


#Test on a different image
#READ EXTERNAL IMAGE...
#test_img = cv2.imread('C:\\Users\\User\\Desktop\\Jupyter\\rooftop_dataset\\mini_data\\LC_AP_37607046_078.tif', cv2.IMREAD_COLOR)       
#test_img = cv2.imread('LC_AP_37705083_001.tif', cv2.IMREAD_COLOR)
test_img = cv2.imread('CAU.jpg', cv2.IMREAD_COLOR)

test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
#test_img = np.reshape(test_img,(512,512,1))
plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)


# In[23]:


prediction_image1 = prediction.reshape(512,512,1)
plt.imshow(prediction_image1, cmap='gray')


# In[26]:


prediction_image1 = prediction.reshape(512,512,1)
prediction_image1 = prediction_image1 > 0.5
prediction_image1 = prediction_image1 * 255.
plt.imshow(prediction_image1, cmap='gray')

cv2.imwrite('CAU_mask.jpg', prediction_image1)


# In[27]:


#Test on a different image
#READ EXTERNAL IMAGE...
#test_img = cv2.imread('C:\\Users\\User\\Desktop\\Jupyter\\rooftop_dataset\\mini_data\\LC_AP_37607046_078.tif', cv2.IMREAD_COLOR)       
#test_img = cv2.imread('LC_AP_37705083_001.tif', cv2.IMREAD_COLOR)
test_img = cv2.imread('AnyConv.com__croped.tif', cv2.IMREAD_COLOR)

test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
#test_img = np.reshape(test_img,(512,512,1))
plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)


# In[28]:


prediction_image1 = prediction.reshape(512,512,1)
plt.imshow(prediction_image1, cmap='gray')


# In[29]:


prediction_image1 = prediction.reshape(512,512,1)
prediction_image1 = prediction_image1 > 0.5
prediction_image1 = prediction_image1 * 255.
plt.imshow(prediction_image1, cmap='gray')

#cv2.imwrite('CAU_mask.jpg', prediction_image1)


# In[25]:


#Test on a different image
#READ EXTERNAL IMAGE...
#test_img = cv2.imread('C:\\Users\\User\\Desktop\\Jupyter\\rooftop_dataset\\mini_data\\LC_AP_37607046_078.tif', cv2.IMREAD_COLOR)       
#test_img = cv2.imread('LC_AP_37705083_001.tif', cv2.IMREAD_COLOR)
test_img = cv2.imread('dongjak_close.jpg', cv2.IMREAD_COLOR)

test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
#test_img = np.reshape(test_img,(512,512,1))
plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)


# In[26]:


prediction_image1 = prediction.reshape(512,512,1)
plt.imshow(prediction_image1, cmap='gray')


# In[ ]:


prediction_image1 = prediction.reshape(512,512,1)
prediction_image1 = prediction_image1 > 0.6
prediction_image1 = prediction_image1 * 255.
plt.imshow(prediction_image1, cmap='gray')


# In[ ]:




