import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import csv
import cv2

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#import matplotlib.pyplot as plt
#%matplotlib inline


#
# Helper functions
#

def load_rgb_image(path_to_image):
    """
    Helper function to load an RGB image (opencv loads images as BGR).
    """
    return cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)

def load_sample(path_to_image, steering_angle, add_flipped=False, load_images=True):
    """
    Retrieves the image if specified and returns the information
    as a list of tuples (image_path, steering_angle, is_flipped, image_data)
    
    If add_flipped is True, the image is flipped and an additional tuple
    is returned with -steering_angle
    
    If load_images is True, then the image is loaded and stored
    as the fourth entry in the tuple.  If this is False, then the image
    is not loaded and the fourth entry is set to None.
    """
    samples = list()
    image = None
    
    if load_images:
        image = load_rgb_image(path_to_image)
    samples.append((path_to_image, steering_angle, False, image))

    # Add the center image (flipped)
    if image is not None:
        image = np.fliplr(image)
    samples.append((path_to_image, -steering_angle, True, image))
    
    return samples

#
# Hyperparameters
#
# Total number of training epochs
# Actual number of epochs is chosen at the end
TRAINING_EPOCHS = 15
BATCH_SIZE = 32

# Training/validation split
# This is the fraction of the data to be used for validation
VALIDATION_SPLIT = 0.2

# This is the amount of adjustment used for left/right cameras
STEERING_ADJUSTMENT = 0.3


#
# Data loading
#

# specify the paths to the image data
data_paths = [
               "./track-1/",            # Track driven in the original direction
               "./track-1-reverse/",    # Track driven in the reverse direction
               "./recenter/",           # Recenter if pointing at the edge
             ]

input_samples = list()

#
# Load up all the directories, this includes reversing of images
# to add more samples
#
# Input CSV data format
# column 0 : path to center camera image
# column 1 : path to left camera image
# column 2 : path to right camera image
# column 3 : steering angle
# column 4 : ?
# column 5 : ?
# column 6 : ?
#

for data_path in data_paths:
    lines = []
    
    # open up each directory, look for the csv file and read it all in
    with open(data_path+"driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    # for each line in the csv file, fixup the paths and add the image
    for line in lines:
        #
        # input_samples tuple format
        # 0 : path to image
        # 1 : steering angle
        # 2 : is the image a flipped image?
        # 3 : the actual image (set to None if not enough memory)
        #     if this is None, the image will be loaded and flipped on demand
        #
        steering_angle = float(line[3])
        data_dir = data_path + "IMG/"

        # Get the filename for the center image
        filename = line[0].split('/')[-1]
        
        # Add the center image (and its flipped image)
        input_samples.extend(load_sample(data_dir + filename,
                                         steering_angle,
                                         add_flipped=True,
                                         load_images=False))
        
        # Get the filename for the left image
        filename = line[1].split('/')[-1]
        
        # Add the left image (and its flipped image)
        input_samples.extend(load_sample(data_dir + filename,
                                         steering_angle + STEERING_ADJUSTMENT,
                                         add_flipped=True,
                                         load_images=False))

        # Get the filename for the right image
        filename = line[2].split('/')[-1]
        
        # Add the right image (and its flipped image)
        input_samples.extend(load_sample(data_dir + filename,
                                         steering_angle - STEERING_ADJUSTMENT,
                                         add_flipped=True,
                                         load_images=False))


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_path, center_angle, is_flipped, image = batch_sample

                if image is None:
                    image = load_rgb_image(image_path)
                    if is_flipped:
                        image = np.fliplr(image)
                
                images.append(image)
                angles.append(center_angle)

            # Note: normalization/truncation is done as part of the
            # model (since it also needs to be done to the input image)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_samples, validation_samples = train_test_split(input_samples, test_size=VALIDATION_SPLIT)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


def build_model():
    model = Sequential()

    # Incoming images are : 160x320x3
    # Crop 60 pixels off the top and 20 pixels off the bottom
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))

    # Normalize the image
    # scale the image to (-1,+1)
    model.add(Lambda(lambda x: (x/128.0) - 1.0, input_shape=(80,320,3)))

    # There are three convolutional layers
    # with a 5x5 kernel and a stride of 2x2
    # all with relu
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    
    # Then there are two convolution layers
    # with a 3x3 kernel and no striding
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    
    # Final fully-connected layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


# Pick the number of epochs from above, train the model, then save it
model = build_model()
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=11,
                    verbose=1)

model.save('model.h5')
print("Model saved to model.h5")

