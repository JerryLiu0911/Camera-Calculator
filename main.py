# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt


def rand_image_generate(h, b):
    input_image = np.random.rand(h, b, 3)
    return input_image


def show_image(input_image):
    plt.figure()
    plt.imshow(input_image)
    plt.show()


def padding(input_image, padding=1):
    ny = input_image.shape[0]
    nx = input_image.shape[1]
    size = input_image.shape[2]
    padded_image = np.ones((nx + padding * 2, ny + padding * 2, size))
    padded_image[padding:-padding, padding:-padding, :] = input_image
    return padded_image


if __name__ == '__main__':
    black_image = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    input_image = rand_image_generate(400, 400)
    print(input_image)
    show_image(input_image)
    show_image(padding(input_image,50))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
