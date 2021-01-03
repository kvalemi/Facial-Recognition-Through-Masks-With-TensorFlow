## Load packages and dependancies ##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import sys



## Utility Functions

# Function for applying transformations
def apply_image_trans(img):
    img_size = 224
    final_image = cv2.resize(img, (img_size, img_size))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0
    
    return final_image

# Function for interpreting the sigmoid output
def convert_sigmoid_output(sigmoid):
    if(sigmoid < 0.50):
        print('\nThis person is Kaveh!\n')
    elif(sigmoid > 0.50):
        print('\nThis person is not Kaveh!\n')
    else:
        print('\nNot Sure!\n')



## Starting Point ##

# Upload the model (if you don't want to train it)
model = keras.models.load_model('./Kaveh_With_Mask_Detection.h5')

## some variables we will need for the Web Cam detection

# font size of detection box
font_scale = 5

# font type of detection text
font = cv2.FONT_HERSHEY_PLAIN

# Coverting to image size that is required by ImageNet (224 x 224)
img_size = 224

# input image path
img_path = sys.argv[1]

# Load image
img = cv2.imread(img_path)

# obtain transformed face
final_image = apply_image_trans(img)

# this is the output of the model
Prediction = model.predict(final_image)
convert_sigmoid_output(Prediction)        

# Tag the face on the image
if(Prediction > 0.50):
    status = "Not Kaveh!"   
elif(Prediction < 0.50):
    status = "Kaveh!"
else:
    status = "Not Sure!"


# Add the text
cv2.putText(img, status, (0, 50), font, font_scale, (255,0,0), 2)


# show image
# cv2.imwrite('./Kaveh_Detection_Outcome.png', img)
