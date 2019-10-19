import matplotlib.pyplot as plt
import pydicom
import numpy as np


# function for viewing dicom images


def visualize_dicom(filename, colormap):
    ds = pydicom.dcmread(filename)
    plt.imshow(ds.pixel_array, cmap=colormap)
    plt.show()


# function for HU Transformation
# Formula -> HU = Gray_Value * slope + intercept
# The latter fields are normally recorded as 1 (slope) and -1024 (intercept) and can be found in the metadata.


def apply_hounsfield_transformation(filename):
    dicom = pydicom.read_file(filename)
    image = dicom.pixel_array.astype(np.float64)

    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope

    if slope is not 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.float64)

    image += np.float64(intercept)

    # Setting values smaller than air, to air. Values smaller than -1024, are probably just outside the scanner.
    image[image < -1024] = -1024
    return image


# code to test
if __name__ == '__main__':
    filename = r"../images/ID_0000ca2f6.dcm"
    visualize_dicom(filename, plt.cm.bone)
    converted_image = apply_hounsfield_transformation(filename)
    plt.imshow(converted_image, cmap=plt.cm.bone)
    plt.show()
