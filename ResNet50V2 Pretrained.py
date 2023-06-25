import numpy as np
import tensorflow as tf
import random
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, cohen_kappa_score
from keras.applications import ResNet50V2

categories = ['benign', 'malignant', 'normal']
directory = 'Full Dataset/'

#importing full dataset
data = []
img_size = 224
for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (img_size, img_size))
        data.append([new_array, class_num])
random.shuffle(data)

#features, label, and train_test_split
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X / X.max()
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

#Data generators and flow
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_gen = train_datagen.flow(
    X_train,
    y_train,
    batch_size=32,
    shuffle=True,
)
test_gen = test_datagen.flow(
    X_test,
    y_test,
    batch_size=32,
    shuffle=True,
)

#constructing model
basemodel=ResNet50V2(include_top=False, weights='imagenet')
basemodel.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = basemodel(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(3, activation = 'softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_gen, epochs=3, validation_data=test_gen)

#plotting results
y_pred = model.predict(X_test, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test_arg, y_pred_bool))
print("Confusion Matrix:")
print(confusion_matrix(y_true=y_test_arg, y_pred=y_pred_bool))
print("Cohen_kappa coefficient;", cohen_kappa_score(y_test_arg, y_pred_bool))

plt.plot(history.history['accuracy'], label='Train_accuracy')
plt.plot(history.history['val_accuracy'], label='Validation_accuracy')
plt.plot(history.history['loss'], label='Train_loss')
plt.plot(history.history['val_loss'], label='Validation_loss')
plt.title('Model Accuracy/Loss')
plt.ylabel('Accuracy/Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
