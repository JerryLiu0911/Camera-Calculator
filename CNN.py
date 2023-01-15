import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pathlib

data_dir = pathlib.Path('C:/Users/jerry/OneDrive/Documents/GitHub/Camera-Calculator/dataset')
image_count = len(list(data_dir.glob('*/*.*')))
print("Total no of images =", image_count)
# print("image size = ",data_dir.glob('*/*.*').shape)
# img_shape = data_dir.glob('*/*.*').shape
# splitting data
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    # color_mode="grayscale",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(50, 50),
    color_mode='grayscale'
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(50, 50),
    color_mode='grayscale'
)

print(train_ds.class_names)
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(np.shape(x_train))
# x_train = np.array([cv2.resize(img, (50,50)) for img in x_train ])
# x_test = np.array([cv2.resize(img, (50,50)) for img in x_test ])
# x_train = x_train / 255
# x_test = x_test / 255

# x_train_flattened = x_train.reshape(len(x_train),784)
# print(x_train_flattened.shape)
# x_test_flattened = x_test.reshape(len(x_test),784)

# Create the model
model = keras.Sequential([
    # keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # keras.layers.MaxPooling2D((2, 2)),
    # keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # keras.layers.MaxPooling2D((2, 2)),
    # keras.layers.Flatten(),
    # keras.layers.Dense(10, activation='softmax')
    keras.layers.Rescaling(1.0 / 255),
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 32)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ds, validation_data= val_ds, epochs=20)
model.save('CNN_symbols')

# model = keras.models.load_model('CNN')
# y_predicted = model.predict(x_test_flattened)
# print(np.argmax(y_predicted[0]))
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
#
#
# class kernel:
#     def __init__(self, W, b, hparameters):
#         self.W = W  # set weights
#         self.b = b  # set biases
#         self.hparameters = hparameters  # set size of kernel
#
#
# def random_image_gen(h, w):
#     rand_image = np.random.rand(h, w, 3)
#     return rand_image
#
#
# def zero_padding(image, pad):
#     x = image.shape[0]
#     y = image.shape[1]
#     if pad > 0:
#         padded_image = np.zeros((x + pad * 2, y + pad * 2, 1))
#         padded_image[pad:-pad, pad:-pad, :] = image
#     else:
#         padded_image = image
#
#     return padded_image
#
#
#     # image is an array of shape (h,w,c) , where c is the rgb channel
#     # padding is the size of padding around the corners
#     padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
#     # np.pad returns a np.array with padded values (defined by constant_values) onto the image (the first parameter).
#     # the actual change in dimensions of the input image is defined by the vectors in the second parameter
#     # in this scenario the output image will have size of (h+padding, w+padding, c)
#     # note that the input should be a batch of output images (m,h,w,c) where m is the number of images in the batch
#     return padded_image
#
#
# def single_conv_layer(input_layer, kernel):
#     stride = kernel.hparameters['stride']
#     pad = kernel.hparamters['pad']
#     m, n_H_prev, n_W_prev, n_C_prev = input_layer.shape
#     # m = number of images in the input batch
#
#
# def show_image(image):
#     plt.figure()
#     plt.imshow(image)
#     plt.show()
#
#
# if __name__ == '__main__':
#     pad = 2
#     image = np.asarray(plt.imread('math1.jpg'))
#     show_image(zero_padding(image, pad))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
