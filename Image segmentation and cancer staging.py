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



#loading data and preprocessing
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
X_train_dataset, X_test_dataset, y_train_dataset, y_test_dataset = train_test_split(input_image, input_mask, random_state=1000) #train test split
X_train_dataset = X_train_dataset.reshape(X_train_dataset.shape[0], img_size, img_size, 3)
X_test_dataset = X_test_dataset.reshape(X_test_dataset.shape[0], img_size, img_size, 3)
y_train_dataset = y_train_dataset.reshape(y_train_dataset.shape[0], img_size, img_size, 3)
y_test_dataset = y_test_dataset.reshape(y_test_dataset.shape[0], img_size, img_size, 3)




#image data generator and flow
seed=1337

train_datagen = ImageDataGenerator(rotation_range=360,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   horizontal_flip=True,
                                   vertical_flip=True
                                   )
test_datagen = ImageDataGenerator(rotation_range=360,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   horizontal_flip=True,
                                   vertical_flip=True
                                  )

train_datagen.fit(X_train_dataset, augment=True, seed=seed)
test_datagen.fit(X_test_dataset, augment=True, seed=seed)

train_gen = train_datagen.flow(
    X_train_dataset,
    y_train_dataset,
    batch_size=1024,
    shuffle=True,
    seed=seed
)
test_gen = test_datagen.flow(
    X_test_dataset,
    y_test_dataset,
    batch_size=1024,
    shuffle=True,
    seed=seed
)



#creating model
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
   inputs = layers.Input(shape=(256,256,3))

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


#get model
model = build_unet_model()


#complie and summary
model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics="accuracy")

model.summary()



#train model
model_history = model.fit(train_gen,
                              epochs=3,
                              validation_data=test_gen)
model.save('image_segmentation.h5')



#mask prediction
model = load_model('image_segmentation.h5')
predicted_mask = model.predict(X_test_dataset)
plt.imsave('predicted_mask.png', predicted_mask)

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
