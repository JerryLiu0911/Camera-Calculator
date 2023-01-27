import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import itertools
import ImageSegmentationModule as sg

# testing with image data
# image, areas, aspect_ratios = sg.segment('Images/math3.jpg',size = 50, Contour_thresh=20, test=True)
#
# # converts to input
# img_array = keras.preprocessing.image.img_to_array(image)
#
# # load pre-trained model from CNN.py
model = keras.models.load_model('CNN')
#
# # outputs the 'confidence' of each classification, as well as the corresponding index to the class.
# prediction = model.predict(img_array)
# print(prediction)
# print([max(p) for p in prediction])
# prediction = prediction.argmax(axis=1)
classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/', '=', '*', '-', 'x']
# ans = [classNames[i] for i in prediction]
# print(ans)
symbols = ['+', '/', '=', '*', '-']

# Context assisted classification : after taking in the predicted values from the neural network, factors such as
# size of the cropping box and mathematical syntax are taken into account to correct misclassified symbols.
# for i in range(0, len(ans)):
#     if (ans[i] == 'mul' and ans[i + 1] in symbols) or (
#             ans[i] == '4' and areas[i] <= max(areas) / 1.8 and 0.75<aspect_ratios[i]<1.2):
#         ans[i] = 'x'
#     if (ans[i] == '8' or ans[i] == '2') and areas[i] <= np.mean(areas) / 3:
#         ans[i] = 'eq'
#     if (ans[i] == '6' and areas[i] <= np.mean(areas) / 2 and 0.8<aspect_ratios[i]<1.2):
#         ans[i]='+'
# print(aspect_ratios)
# print(ans)
# print(areas)
# print(np.mean(areas))
# print(max(areas))

# Load the training dataset
data_dir = pathlib.Path('C:/Users/jerry/OneDrive/Documents/GitHub/Camera-Calculator/dataset')
# splitting data
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=50,
    image_size=(50, 50),
    color_mode='grayscale',
    shuffle=True
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=50,
    image_size=(50, 50),
    color_mode='grayscale',
    shuffle=True
)

# Converts the test_ds to a numpy array
test_ds = val_ds.as_numpy_iterator()

y_pred = []  # store predicted labels
y_true = []  # store true labels

# iterate over the dataset to concatenate the predicted and truth labels to a
# single dimensional array for the confusion matrix funcion.
for image_batch, label_batch in val_ds:
    # append true labels
    y_true.append(label_batch)
    # compute predictions
    preds = model.predict(image_batch)
    # append predicted labels
    y_pred.append(np.argmax(preds, axis=- 1))

correct_labels = tf.concat([item for item in y_true], axis=0)
predicted_labels = tf.concat([item for item in y_pred], axis=0)


def ConfusionMatrix(predicted_labels, true_labels, classNames):
    # creates confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # using a colour map to visualize the confusion matrix
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)

    # converting values within the confusion matrix to percentages of accuracy

    txt = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    fmt = '.3f'
    thresh = cm.max() / 2.

    # changing font colour depending on the background colour
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(txt[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


figure = ConfusionMatrix(predicted_labels, correct_labels, classNames)
plt.show()
