import PIL
import tensorflow as tf
from keras.saving.legacy.save import load_model
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

image_dir = 'D:\Mathieu\Program Files\AARD Research\Full Dataset\Image segmentation data\image'
mask_dir = 'D:\Mathieu\Program Files\AARD Research\Full Dataset\Image segmentation data\masks'
img_size = 256

categories = ['Image', 'Annotations']
input_image = []
for file in os.listdir(image_dir):
   filepath = os.path.join(image_dir, file)
   img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
   new_array = cv2.resize(img_array, (img_size, img_size))
   input_image.append([new_array])

input_mask = []
for file in os.listdir(mask_dir):
   filepath = os.path.join(mask_dir, file)
   img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
   new_array = cv2.resize(img_array, (img_size, img_size))
   input_mask.append([new_array])

input_image = np.array(input_image).reshape(-1, img_size, img_size, 3)
input_image = input_image / input_image.max()
input_mask = np.array(input_mask).reshape(-1, img_size, img_size, 3)
input_mask = input_mask / input_mask.max()

seed=1337

image_datagen = ImageDataGenerator(rotation_range=360,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   horizontal_flip=True,
                                   vertical_flip=True
                                   )
mask_datagen = ImageDataGenerator(rotation_range=360,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   horizontal_flip=True,
                                   vertical_flip=True
                                  )

image_datagen.fit(input_image, augment=True, seed=seed)
mask_datagen.fit(input_mask, augment=True, seed=seed)

image_gen = image_datagen.flow(
    input_image,
    batch_size=32,
    shuffle=True,
    seed=seed
)
mask_gen = mask_datagen.flow(
    input_mask,
    batch_size=32,
    shuffle=True,
    seed=seed
)
train_gen = zip(image_gen, mask_gen)

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D((2,2), padding='same')(f)
   p = layers.Dropout(0.3)(p)

   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)

   return x

def build_unet_model():
 # inputs
   inputs = keras.Input(shape = input_image.shape[1:])

   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)

   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)

   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)

   # outputs
   outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

   return unet_model

model = build_unet_model()

model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics="accuracy")

model.summary()

model_history = model.fit(train_gen, epochs=1)
model.save('image_segmentation.h5')

model = load_model('image_segmentation.h5')
predicted_mask = model.predict(input_image)
plt.imsave('predicted_mask.png', predicted_mask[0])


plt.subplot(131)
plt.title('image')
plt.imshow(input_image[0])

plt.subplot(132)
plt.title('predicted_mask')
plt.imshow(cv2.imread('predicted_mask.png'))

plt.subplot(133)
plt.title('true mask')
plt.imshow(input_mask[0])
plt.savefig('image_segmentation_result')
plt.show()
