import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import itertools
import ImageSegmentationModule as sg



#testing with image data
image = sg.segment('croppedinput.jpg')
image = np.array(image)/255
model = keras.models.load_model('CNN')
print(model.predict(image).argmax(axis = 1))

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the input images
#x_test = np.array([cv2.resize(img, (50,50)) for img in x_test ])
x_test = x_test / 255.0

for i in range(0, 10):
    cv2.imshow('test', cv2.resize(x_test[i], (300, 300)))
    cv2.waitKey(0)
model = keras.models.load_model('CNN')
print(y_test.shape)
predictions = model.predict(x_test)
labels = predictions.argmax(axis = 1)
print(labels.shape)
def ConfusionMatrix (labels, classNames, normalize = False):
    cm = confusion_matrix(y_test, labels)
    figure = plt.figure(figsize = (8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    txt = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(txt[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

figure = ConfusionMatrix(labels, range(0,10))
plt.show()