# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:08:29 2020

@author: dhruv
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator

Classifier = Sequential()

## conv2d with 32 filters , 3x3 filter size strides = 1x1 
Classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3)),activation='relu')


Classifier.add(MaxPooling2D(pool_size=(2,2),))

Classifier.add(Flatten())

Classifier.add(Dense(units=128,activation='relu'))
Classifier.add(Dense(units=1,activation='sigmoid'))


Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])




train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2 ,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

Classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)    
