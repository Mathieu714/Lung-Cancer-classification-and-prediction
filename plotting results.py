import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#SMOTE Bar-graph
X_train_smt = np.load('X_train_smt_GRAYSCALE.npy')
y_train_smt = np.load('y_train_smt.npy')
X_test_smt = np.load('X_test_smt_GRAYSCALE.npy')
y_test_smt = np.load('y_test_smt.npy')

X_train = np.load('X_train_GRAYSCALE.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test_GRAYSCALE.npy')
y_test = np.load('y_test.npy')

name = ['benign', 'malignant', 'normal']
d = dict(Counter(y_train))
d_smt = dict(Counter(y_train_smt))
value = [d[0], d[1], d[2]]
value_smt = [d_smt[0], d_smt[1], d_smt[2]]

fig, axes = plt.subplots(1,2)
plt.subplot(121)
plt.title('Before SMOTE')
for x, y in zip(name, value):
    plt.bar(x, y)
    plt.text(x, y, '%d' % (int(y)),
             horizontalalignment='center',
             verticalalignment='bottom')

plt.subplot(122)
plt.title('After SMOTE')
for x, y in zip(name, value_smt):
    plt.bar(x, y)
    plt.text(x, y, '%d' % (int(y)),
             horizontalalignment='center',
             verticalalignment='bottom')

fig.supxlabel('Category')
fig.supylabel('# of image')
plt.savefig('SMOTE Bar Graph.jpg')
plt.show()
