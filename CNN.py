import matplotlib.pyplot as plt
import numpy as np
import cv2


class kernel:
    def __init__(self, W, b, hparameters):
        self.W = W  # set weights
        self.b = b  # set biases
        self.hparameters = hparameters  # set size of kernel


def random_image_gen(h, w):
    rand_image = np.random.rand(h, w, 3)
    return rand_image


def zero_padding(image, pad):
    x = image.shape[0]
    y = image.shape[1]
    if pad > 0:
        padded_image = np.zeros((x + pad * 2, y + pad * 2, 1))
        padded_image[pad:-pad, pad:-pad, :] = image
    else:
        padded_image = image

    return padded_image


    # image is an array of shape (h,w,c) , where c is the rgb channel
    # padding is the size of padding around the corners
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    # np.pad returns a np.array with padded values (defined by constant_values) onto the image (the first parameter).
    # the actual change in dimensions of the input image is defined by the vectors in the second parameter
    # in this scenario the output image will have size of (h+padding, w+padding, c)
    # note that the input should be a batch of output images (m,h,w,c) where m is the number of images in the batch
    return padded_image


def single_conv_layer(input_layer, kernel):
    stride = kernel.hparameters['stride']
    pad = kernel.hparamters['pad']
    m, n_H_prev, n_W_prev, n_C_prev = input_layer.shape
    # m = number of images in the input batch


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pad = 2
    image = np.asarray(plt.imread('math1.jpg'))
    show_image(zero_padding(image, pad))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
