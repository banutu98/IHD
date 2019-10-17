import matplotlib.pyplot as plt
import pydicom
import numpy as np

# function for viewing dicom images


def visualize_dicom(images_path, image_id, colormap):
    filename = create_filename(images_path, image_id)
    ds = pydicom.dcmread(filename)
    plt.imshow(ds.pixel_array, cmap=colormap)
    plt.show()

# function for HU Transformation
# Formula -> HU = Gray_Value * slope + intercept
# The latter fields are normally recorded as 1 (slope) and -1024 (intercept) and can be found in the metadata.


def apply_hounsfield_transformation(images_path, image_id):
    dicom = pydicom.read_file(create_filename(images_path, image_id))
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


def create_filename(images_path, image_id):
    return images_path + 'ID_' + image_id + '.dcm'


# code to test
if __name__ == '__main__':
    images_path = r"../images/"
    image_id = "0000ca2f6"
    visualize_dicom(images_path, image_id, plt.cm.bone)
    converted_image = apply_hounsfield_transformation(images_path, image_id)
    plt.imshow(converted_image, cmap=plt.cm.bone)
    plt.show()
