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


## Utility Functions ##

# obtain training data function
def create_training_data():
    
    faceCascade = cv2.CascadeClassifier('./haarcascade_face.xml')

    # Loop through training class folders
    for category in Classes:

        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category) # Class Labels

        # Loop through images in each folder
        for img in os.listdir(path):

            # Add images to training set with correct labels
            try:
                img_array = cv2.imread(os.path.join(path, img))

                # apply the necessary transformations
                img_size = 224
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array, class_num])

            except Exception as e:
                #print('Failed to Grab Image')
                pass



## Data Pre-Processing ##

# data Directory
Datadirectory = "Dataset/"

# Define labels
Classes = ["Kaveh", "Not_Kaveh"]

# Coverting to image size that is required by ImageNet (224 x 224)
img_size = 224

# Training data
training_Data = []

# Load training data
create_training_data()

# shuffle the training data
random.shuffle(training_Data)

# Convert to Array
X = [] # Features
y = [] # label

for features, label in training_Data:
    X.append(features)
    y.append(label)
    
# Convert to Numpy Array
X = np.array(X).reshape(-1, img_size, img_size, 3)
Y = np.array(y)
    
# Normalize the features
X = X/255



## Begin Training With The 

# Transfer Learning - Load lightweight pre-trained DL model
model = tf.keras.applications.mobilenet.MobileNet()
# model.summary()

# Obtain transfer model layer input and output
base_input = model.layers[0].input
base_ouput = model.layers[-4].output

# Modify transfer model's last layer
Flat_layer = layers.Flatten()(base_ouput)
final_output = layers.Dense(1)(Flat_layer)
# Add sigmoid since this is a binary classification
final_output = layers.Activation('sigmoid')(final_output)

# Create the revised model
new_model = keras.Model(inputs = base_input, outputs = final_output)
new_model.summary()

# Set up the model
new_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit the model to the data
epochs = 20
new_model.fit(X, Y, epochs = epochs, validation_split = 0.1)


# Save Our Model
new_model.save("Kaveh_With_Mask_Detection.h5")
