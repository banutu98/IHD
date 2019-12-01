import copy

import pydicom
import scipy
from skimage import morphology
from skimage.transform import resize
from utilities.augmentations import *


class Preprocessor:

    @staticmethod
    def apply_hounsfield(image, intercept, slope):
        if slope is not 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.float64)

        image += np.float64(intercept)

        # Setting values smaller than air, to air. Values smaller than -1024, are probably just outside the scanner.
        image[image < -1024] = -1024
        return image

    @staticmethod
    def windowing(image, custom_center=30, custom_width=100, rescale=True):
        new_image = copy.deepcopy(image)
        min_value = custom_center - (custom_width / 2)
        max_value = custom_center + (custom_width / 2)

        # Including another value for values way outside the range, to (hopefully) make segmentation processes easier.
        new_image[new_image < min_value] = min_value
        new_image[new_image > max_value] = max_value
        if rescale:
            new_image = (new_image - min_value) / (max_value - min_value)
        return new_image

    @staticmethod
    def image_resample(image, pixel_spacing, new_spacing=[1, 1]):
        pixel_spacing = map(float, pixel_spacing)
        spacing = np.array(list(pixel_spacing))
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        return image

    @staticmethod
    def image_background_segmentation(image, WL=30, WW=100, rescale=True):
        lB = WW - WL
        uB = WW + WL

        # Keep only values inside of the window
        background_separation = np.logical_and(image > lB, image < uB)

        # Get largest connected component:
        # From https://github.com/nilearn/nilearn/blob/master/nilearn/_utils/ndimage.py
        background_separation = morphology.dilation(background_separation, np.ones((5, 5)))
        labels, label_nb = scipy.ndimage.label(background_separation)

        label_count = np.bincount(labels.ravel().astype(np.int))
        # discard the 0 label
        label_count[0] = 0
        mask = labels == label_count.argmax()

        # Fill holes in the mask
        mask = morphology.dilation(mask, np.ones((5, 5)))  # dilate the mask for less fuzy edges
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))  # dilate the mask again

        image = mask * image

        if rescale:
            img_min = np.amin(image)
            img_max = np.amax(image)
            image = (image - img_min) / (img_max - img_min)
        return image

    @staticmethod
    def preprocess(image_path):
        dicom = pydicom.read_file(image_path)
        image = dicom.pixel_array.astype(np.float64)
        if image.shape != (512, 512):
            image = resize(image, (512, 512))
        p = Preprocessor
        image = p.apply_hounsfield(image, dicom.RescaleIntercept, dicom.RescaleSlope)
        image = p.windowing(image)
        return image

    @staticmethod
    def augment(image):
        augmented = list()
        augmented.append(blur_image(image))
        augmented.append(noisy(image))
        augmented.append(adjust_brightness(image, 0.3))
        return augmented


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = Preprocessor.preprocess(r'data\ID_0000aee4b.dcm')
    Preprocessor.augment(image)
    plt.imshow(image, cmap=plt.cm.get_cmap('bone'))
    plt.savefig('test.png')
