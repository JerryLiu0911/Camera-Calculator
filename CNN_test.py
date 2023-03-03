import tensorflow as tf
from keras import Model
from tensorflow import keras
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import ImageSegmentationModule as sg


''' Testing with image data '''

image, areas, aspect_ratios = sg.segment('croppedinput.jpg', size=50, test=True)


# converts to input
img_array = np.reshape(image, [len(image), 50, 50, 1])
img_array = np.array(img_array, np.float32)
print(img_array.shape)

# load converted tflite file
interpreter = tf.lite.Interpreter(model_path='CNN.tflite')
interpreter.allocate_tensors()



# input_details is stored in the form of a dictionary, therefore we can access the input shape of the neural network
# by calling the first layer of the neural network (index 0) with the parameter name "shape"
input_shape = interpreter.get_input_details()[0]["shape"]
print(input_shape)

predictions = []
certainty = []
for input in img_array:
    input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input)   # configuring input to neural network from image
    interpreter.invoke()                                                            # Instead of model.predict, invoke() is used.
    prediction = interpreter.get_tensor( interpreter.get_output_details()[0]['index'])
    predictions.append(prediction)

# Reshapes the output predictions to a 2d array for ease of processing.
predictions = np.reshape(predictions, (len(predictions), 16))

# Collects the indexes of the highest certainty
predictedIndex = predictions.argmax(axis = 1)
certainty = [max(p) for p in predictions]

print("predictedIndex", predictedIndex)
print("calculated certainty", certainty)

classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '÷', '=', '*', '-', 'x']
ans = [classNames[i] for i in predictedIndex]
print(ans)

model = keras.models.load_model("CNN")

''' Plot feature map of first conv layer for given image '''

# redefine model to output right after the first hidden layer
feature_map = Model(inputs=model.inputs, outputs=model.layers[1].output)
# retrieve feature map for first hidden layer
feature_maps = feature_map.predict(img_array)
# Plots the
square = 3
counter = 1
for i in range(square):
    for i in range(2):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, counter)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        print(counter)
        plt.imshow(feature_maps[0, :, :, counter - 1], cmap='gray')
        counter += 1

plt.show()


'''Calculating confusion matrix'''

# Load the training dataset
data_dir = pathlib.Path('C:/Users/jerry/OneDrive/Documents/GitHub/Camera-Calculator/dataset')
# splitting data
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
