import cv2
import os
import pydicom
from utilities import Window
import matplotlib.pyplot as plt
import numpy as np

KERNEL_WIDTH = 13
KERNEL_HEIGHT = 13
GAUSS_MEAN = 0.5
GAUSS_STDDEV = 0.05
BRIGHTNESS_DELTA = 0.4

# the kernel sizes must be positive odd integers but they do not have to be equal
# the larger they are the more the image will be blurred


def blur_image(pixel_matrix, kernel_size_width, kernel_size_height):
    return cv2.GaussianBlur(pixel_matrix, (kernel_size_width, kernel_size_height), cv2.BORDER_DEFAULT)


def noisy(image, mean, stddev):
    gauss = np.random.normal(mean, stddev, image.shape)
    noisy = image + gauss
    noisy_min = np.amin(noisy)
    noisy_max = np.amax(noisy)
    noisy = (noisy - noisy_min) / (noisy_max - noisy_min)
    return noisy


def adjust_brightness(image, delta):
    image += delta
    image[image < 0] = 0
    image[image > 1] = 1
    return image


# code to test
if __name__ == '__main__':
    case = os.path.join('..', 'images', 'ID_000012eaf.dcm')

    data = pydicom.read_file(case)

    intercept, slope = Window.get_windowing(data)
    img = pydicom.read_file(case).pixel_array

    windowed_img = Window.window_image(img, 600, 2800, intercept, slope)

    blurred_img = blur_image(windowed_img, KERNEL_HEIGHT, KERNEL_WIDTH)

    plt.imshow(windowed_img, cmap=plt.cm.bone)
    plt.show()
    plt.imshow(blurred_img, cmap=plt.cm.bone)
    plt.show()
    plt.imshow(noisy(blurred_img, GAUSS_MEAN, GAUSS_STDDEV), cmap=plt.cm.bone)
    plt.show()
    plt.imshow(adjust_brightness(windowed_img, BRIGHTNESS_DELTA), cmap=plt.cm.bone)
    plt.show()
