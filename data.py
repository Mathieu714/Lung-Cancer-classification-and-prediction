import os
from collections import Counter
import random
import cv2
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

categories = ['benign', 'malignant', 'normal']
directory = 'Full Dataset/'

#importing full dataset
data_GRAYSCALE = []
img_size = 224
for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))
        data_GRAYSCALE.append([new_array, class_num])
random.shuffle(data_GRAYSCALE)

for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        plt.subplot(1,3,class_num+1)
        plt.title(i)
        plt.imshow(img)
        break

plt.savefig('Data examples.jpg')
plt.show()

#features, label, and train_test_split
X_GRAYSCALE = []
y_GRAYSCALE = []
for features, label in data_GRAYSCALE:
    X_GRAYSCALE.append(features)
    y_GRAYSCALE.append(label)
X_GRAYSCALE = np.array(X_GRAYSCALE).reshape(-1, img_size, img_size, 1)
X_GRAYSCALE = X_GRAYSCALE / X_GRAYSCALE.max()
y_GRAYSCALE = np.array(y_GRAYSCALE)
X_train_GRAYSCALE, X_test_GRAYSCALE, y_train_GRAYSCALE, y_test_GRAYSCALE = train_test_split(X_GRAYSCALE, y_GRAYSCALE, random_state=10, stratify=y_GRAYSCALE)
print('before SMOTE')
print(f'y_train_GRAYSCALE:{Counter(y_train_GRAYSCALE)}')
print(f'y_test_GRAYSCALE:{Counter(y_test_GRAYSCALE)}')

np.save('X_train_GRAYSCALE', X_train_GRAYSCALE)
np.save('y_train', y_train_GRAYSCALE)
np.save('X_test_GRAYSCALE', X_test_GRAYSCALE)
np.save('y_test', y_test_GRAYSCALE)

#applying SMOTE
X_train_GRAYSCALE = X_train_GRAYSCALE.reshape(X_train_GRAYSCALE.shape[0], img_size*img_size*1)
X_test_GRAYSCALE = X_test_GRAYSCALE.reshape(X_test_GRAYSCALE.shape[0], img_size*img_size*1)
smt = SMOTE()
X_train_smt_GRAYSCALE, y_train_smt_GRAYSCALE = smt.fit_resample(X_train_GRAYSCALE, y_train_GRAYSCALE)
X_test_smt_GRAYSCALE, y_test_smt_GRAYSCALE = smt.fit_resample(X_test_GRAYSCALE, y_test_GRAYSCALE)
print('after SMOTE')
print(f'y_train_smt_GRAYSCALE:{Counter(y_train_smt_GRAYSCALE)}')
print(f'y_test_smt_GRAYSCALE:{Counter(y_test_smt_GRAYSCALE)}')
X_train_smt_GRAYSCALE = X_train_smt_GRAYSCALE.reshape(X_train_smt_GRAYSCALE.shape[0], img_size, img_size, 1)
X_test_smt_GRAYSCALE = X_test_smt_GRAYSCALE.reshape(X_test_smt_GRAYSCALE.shape[0], img_size, img_size, 1)

np.save('X_train_smt_GRAYSCALE', X_train_smt_GRAYSCALE)
np.save('y_train_smt', y_train_smt_GRAYSCALE)
np.save('X_test_smt_GRAYSCALE', X_test_smt_GRAYSCALE)
np.save('y_test_smt', y_test_smt_GRAYSCALE)

data_COLOR = []
img_size = 224
for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (img_size, img_size))
        data_COLOR.append([new_array, class_num])
random.shuffle(data_COLOR)


#features, label, and train_test_split
X_COLOR = []
y_COLOR = []
for features, label in data_COLOR:
    X_COLOR.append(features)
    y_COLOR.append(label)
X_COLOR = np.array(X_COLOR).reshape(-1, img_size, img_size, 3)
X_COLOR = X_COLOR / X_COLOR.max()
y_COLOR = np.array(y_COLOR)
X_train_COLOR, X_test_COLOR, y_train_COLOR, y_test_COLOR = train_test_split(X_COLOR, y_COLOR, random_state=10, stratify=y_COLOR)
print('before SMOTE')
print(f'y_train_COLOR:{Counter(y_train_COLOR)}')
print(f'y_test_COLOR:{Counter(y_test_COLOR)}')

np.save('X_train_COLOR', X_train_COLOR)
np.save('X_test_COLOR', X_test_COLOR)

#applying SMOTE
X_train_COLOR = X_train_COLOR.reshape(X_train_COLOR.shape[0], img_size*img_size*3)
X_test_COLOR = X_test_COLOR.reshape(X_test_COLOR.shape[0], img_size*img_size*3)
smt = SMOTE()
X_train_smt_COLOR, y_train_smt_COLOR = smt.fit_resample(X_train_COLOR, y_train_COLOR)
X_test_smt_COLOR, y_test_smt_COLOR = smt.fit_resample(X_test_COLOR, y_test_COLOR)
print('after SMOTE')
print(f'y_train_smt_COLOR:{Counter(y_train_smt_COLOR)}')
print(f'y_test_smt_COLOR:{Counter(y_test_smt_COLOR)}')
X_train_smt_COLOR = X_train_smt_COLOR.reshape(X_train_smt_COLOR.shape[0], img_size, img_size, 3)
X_test_smt_COLOR = X_test_smt_COLOR.reshape(X_test_smt_COLOR.shape[0], img_size, img_size, 3)

np.save('X_train_smt_COLOR', X_train_smt_COLOR)
np.save('X_test_smt_COLOR', X_test_smt_COLOR)



