import cv2
import numpy as np
from utilities.defines import *
from mop.mop import *

# the kernel sizes must be positive odd integers but they do not have to be equal
# the larger they are the more the image will be blurred


@blur_monitor
def blur_image(pixel_matrix, kernel_size_width=KERNEL_WIDTH, kernel_size_height=KERNEL_HEIGHT):
    return cv2.GaussianBlur(pixel_matrix, (kernel_size_width, kernel_size_height), cv2.BORDER_DEFAULT)


@noisy_monitor
def noisy(image, mean=GAUSS_MEAN, stddev=GAUSS_STDDEV):
    gauss = np.random.normal(mean, stddev, image.shape)
    noisy = image + gauss
    noisy_min = np.amin(noisy)
    noisy_max = np.amax(noisy)
    noisy = (noisy - noisy_min) / (noisy_max - noisy_min)
    return noisy


@brightness_monitor
def adjust_brightness(image, delta=BRIGHTNESS_DELTA):
    image += delta
    image[image < 0] = 0
    image[image > 1] = 1
    return image


if __name__ == '__main__':
    pass
