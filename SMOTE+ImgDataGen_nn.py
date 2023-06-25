from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, cohen_kappa_score, ConfusionMatrixDisplay


X_train_smt = np.load('X_train_smt_GRAYSCALE.npy')
y_train_smt = np.load('y_train_smt.npy')
X_test_smt = np.load('X_test_smt_GRAYSCALE.npy')
y_test_smt = np.load('y_test_smt.npy')
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
model = Sequential()

model.add(  Conv2D(64, (3,3), input_shape = X_train_smt.shape[1:])    )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(3, activation = 'softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_gen, epochs=1, validation_data=test_gen)

model.save('SimpleModel')

#plotting results
#y_test_arg=np.argmax(y_test_smt, axis=1)
y_pred = model.predict(X_test_smt, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
report = classification_report(y_test_smt, y_pred_bool)
print(report)
cm = confusion_matrix(y_true=y_test_smt, y_pred=y_pred_bool)
print("Confusion Matrix:")
print(cm)
print("Cohen_kappa coefficient:", cohen_kappa_score(y_test_smt, y_pred_bool))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Accuracy/Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training graph')


ConfusionMatrixDisplay(cm).plot()
plt.savefig('cm graph')
plt.show()




