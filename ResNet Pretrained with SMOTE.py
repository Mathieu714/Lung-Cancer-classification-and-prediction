import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, \
    cohen_kappa_score, ConfusionMatrixDisplay

X_train_smt = np.load('D:\Mathieu\Program Files\AARD Research\X_train_smt_COLOR.npy')
y_train_smt = np.load('D:\Mathieu\Program Files\AARD Research\y_train_smt.npy')
X_test_smt = np.load('D:\Mathieu\Program Files\AARD Research\X_test_smt_COLOR.npy')
y_test_smt = np.load('D:\Mathieu\Program Files\AARD Research\y_test_smt.npy')
#Data generators and flow
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_gen = train_datagen.flow(
    X_train_smt,
    y_train_smt,
    batch_size=32,
    shuffle=True,
)
test_gen = test_datagen.flow(
    X_test_smt,
    y_test_smt,
    batch_size=32,
    shuffle=True,
)

#constructing model
basemodel=tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
basemodel.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = basemodel(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(3, activation = 'softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_gen, epochs=1, validation_data=test_gen)

#plotting results
y_pred = model.predict(X_test_smt, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test_smt, y_pred_bool))
print("Confusion Matrix:")
cm = confusion_matrix(y_true=y_test_smt, y_pred=y_pred_bool)
print(cm)
print("Cohen_kappa coefficient;", cohen_kappa_score(y_test_smt, y_pred_bool))

plt.plot(history.history['accuracy'], label='Train_accuracy')
plt.plot(history.history['val_accuracy'], label='Validation_accuracy')
plt.plot(history.history['loss'], label='Train_loss')
plt.plot(history.history['val_loss'], label='Validation_loss')
plt.title('Model Accuracy/Loss')
plt.ylabel('Accuracy/Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training graph ResNet')


ConfusionMatrixDisplay(cm).plot()
plt.savefig('cm graph')
plt.show()
