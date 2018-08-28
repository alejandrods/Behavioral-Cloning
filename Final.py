#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:45:36 2017

@author: Alex
"""

import numpy as np
import random
import cv2
import math
import json
import csv
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.ndimage import rotate
import matplotlib.image as mpimg
from scipy.misc.pilutil import imresize


from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Cropping2D

from random import shuffle
from sklearn.model_selection import train_test_split


##############################
#0. LOAD FILE
##############################

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) 
del(lines[0])

##############################
#1. INITIALIZE IMAGES
##############################
images = []
measurements = []
correction = [0,+0.25,-0.25]
j = 0

for line in lines[:100]:
    print(j)
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])+correction[i]
        measurements.append(measurement)       
        
    j += 1    
    
###############################
##2. AUGMENTATION DATA
###############################

def flip(image, steering):
    flip_image = cv2.flip(image, 1)
    flip_steering = -1 * steering
    
    return flip_image, flip_steering

def random_brightness(image):
    #Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 

def augmentation_function(image, steering):
    
    flip_image, flip_steering = flip(image, steering)
    
    brightness_image = random_brightness(flip_image)
    
    return brightness_image, flip_steering

def augmentation_data(image_data, steering_data, augmentation_number = -1):
    
    if augmentation_number == -1:
        augmentation_number = random.randint(0,len(image_data))
        
    sample_images = random.sample(range(len(image_data)), augmentation_number)
    
    #print(sample_images)
    
    augmented_img=[]
    augmented_steering=[]
    
    for item in sample_images:
        img = image_data[item]
        steering = steering_data[item]
    
        result = augmentation_function(img, steering)
        augmented_img.append(result[0])
        augmented_steering.append(result[1])
        
    return augmented_img, augmented_steering

result = augmentation_data(images, measurements,-1)

images_aug = result[0]
measurements_aug = result[1]

augmented_images = images + images_aug
augmented_measurements = measurements + measurements_aug
print("OK3")

#############################
#3. GENERATE NEW IMAGES
#############################
X_train, X_valid, y_train, y_valid = train_test_split(augmented_images, augmented_measurements, test_size=0.1)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

##############################
#4. MODEL
##############################

#Convolutional NVIDIA

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()
model.compile(optimizer = "adam", loss = "mse")

model.fit(X_train, y_train, validation_data = (X_valid, y_valid), nb_epoch=4, shuffle=True) #, shuffle=True, validation_split=0.1

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model_a2.h5')

