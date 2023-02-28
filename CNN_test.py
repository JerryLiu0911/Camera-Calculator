import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Model
from matplotlib import pyplot
import itertools
import ImageSegmentationModule as sg

# testing with image data
image, areas, aspect_ratios = sg.segment('Images/math3.jpg', size=50, test=False)


# converts to input
img_array = keras.preprocessing.image.img_to_array(image[3])
print(img_array.shape)
img_array = img_array.reshape(1, img_array.shape[0], img_array.shape[1], img_array.shape[2])

# load pre-trained model from CNNTrain.py
model = keras.models.load_model('CNN')
# keras.utils.plot_model(model, "\Images\modelArchitecture.png", show_shapes=True)

# load converted tflite file
interpreter = tf.lite.Interpreter(model_path='CNN.tflite')
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# input_details is stored in the form of a dictionary, therefore we can access the input shape of the neural network
# by calling the first layer of the neural network (index 0) with the parameter name "shape"
input_shape = input_details[0]["shape"]
print(input_shape)

# configuring input to neural network from image
interpreter.set_tensor(input_details[0]['index'], img_array)

# running tflite model
interpreter.invoke()

#retreiving output
prediction = interpreter.get_tensor(output_details[0]['index'])
predictedIndex = prediction.argmax(axis = 1)
print("The predicted answer is ", predictedIndex)

# plot feature map of first conv layer for given image

# redefine model to output right after the first hidden layer
feature_map = Model(inputs=model.inputs, outputs=model.layers[1].output)
# get feature map for first hidden layer
feature_maps = feature_map.predict(img_array)
# plot all 64 maps in an 8x8 squares
square = 3
ix = 1
for i in range(square):
    for i in range(2):
        # specify subplot and turn of axis
        ax = pyplot.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        print(ix)
        pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
        ix += 1
# show the figure
# pyplot.show()
#
# outputs the 'confidence' of each classification, as well as the corresponding index to the class.
prediction = model.predict(img_array)
# print(prediction)
certainty = [max(p) for p in prediction]
# print(certainty)
predictedIndex = prediction.argmax(axis=1)
classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/', '=', '*', '-', 'x']
ans = [classNames[i] for i in predictedIndex]
# print(ans)
symbols = ['+', '/', '=', '*', '-']

# Context assisted classification : after taking in the predicted values from the neural network, factors such as
# size of the cropping box and mathematical syntax are taken into account to correct misclassified symbols.
for i in range(0, len(ans)):
    if ans[i] == '4' and areas[i] <= max(areas) / 1.8 and 0.75 < aspect_ratios[i] < 1.2:
        ans[i] = 'x'
    if (ans[i] == '8' or ans[i] == '2') and areas[i] <= np.mean(areas) / 3:
        ans[i] = '='
    if ans[i] == '6' and areas[i] <= np.mean(areas) / 2 and 0.8 < aspect_ratios[i] < 1.2:
        ans[i] = '='

if ans.count('=') >= 2:
    currentMinEq = 0  # stores the index of the equal sign which has the lowest certainty
    currentMinCert = 2  # stores the certainty of the lowest certain equal sign
    max = 0
    secondMax = 0
    secondMaxIndex = 0
    maxIndex = 0
    for i in range(0, len(ans)):
        if ans[i] == '=':
            if certainty[i]<currentMinCert:
                currentMinEq = i
                currentMinCert = certainty[i]
    for i in range(0, len(symbols)): #Searches for a second possible symbol
        if prediction[currentMinEq][i+9] > max:
            secondMax = max
            max = prediction[currentMinEq][i]
            secondMaxIndex = maxIndex
            maxIndex = i
    print("maxIndex", maxIndex, "secondMaxIndex", secondMaxIndex)
    ans[currentMinEq] = classNames[maxIndex + 9]
    if classNames[maxIndex+9] == '=':
        ans[currentMinEq] = classNames[secondMaxIndex+9]

for k in range(0, len(ans)-1):
    print(k)
    if ans[k] == '*' and ans[k + 1] in symbols:
        ans[k] = 'x'
    elif ans[k] in symbols and ans[k + 1] == '*':
        ans[k + 1] = 'x'
    if ans[k] in symbols and ans[k + 1] in symbols:
        ans = ["Unclear image. Please retake photo."]
        break

print(aspect_ratios)
print(ans)
print(areas)
print(np.mean(areas))
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
# plt.show()
