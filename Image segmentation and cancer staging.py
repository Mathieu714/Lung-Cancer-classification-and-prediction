import os
import random
import cv2
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
import numpy as np

from PIL import ImageOps
from PIL.Image import Image
from PIL._imaging import display
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img

input_dir = 'Full Dataset/Image segmentation data/Image/'
target_dir = 'Full Dataset/Image segmentation data/Annotations/'
num_classes = 3
batch_size = 32
img_size = (160, 160)

categories = ['Image', 'Annotations']
data_input = []
for file in os.listdir(input_dir):
   filepath = os.path.join(input_dir, file)
   img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
   new_array = cv2.resize(img_array, img_size)
   data_input.append([new_array])

data_annotations = []
for file in os.listdir(target_dir):
   filepath = os.path.join(target_dir, file)
   img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
   new_array = cv2.resize(img_array, img_size)
   data_annotations.append([new_array])

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
data_input = np.array(data_input).reshape(-1, 160, 160, 3)
data_annotations = np.array(data_annotations).reshape(-1, 160, 160, 3)

image_datagen.fit(data_input, augment=True, seed=seed)
mask_datagen.fit(data_annotations, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory(
    input_dir,
    class_mode=None,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    target_dir,
    class_mode=None,
    seed=seed)

from keras import layers
import keras

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=(img_size) + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size, num_classes)
model.summary()
# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(data_input)
random.Random(1337).shuffle(data_annotations)
train_input_img_paths = data_input[:-val_samples]
train_target_img_paths = data_annotations[:-val_samples]
val_input_img_paths = data_input[-val_samples:]
val_target_img_paths = data_annotations[-val_samples:]

# Instantiate data Sequences for each split
train_gen = zip(image_generator, mask_generator)

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

# Train the model, doing validation at the end of each epoch.
epochs = 3
model.fit(train_gen, steps_per_epoch=2000, epochs=epochs)

model.save('img_seg_model')
