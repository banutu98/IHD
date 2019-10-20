import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import pydicom

from Hounsfield import apply_hounsfield_transformation

# There are at least 5 windows that a radiologist goes through for each scan!
# Brain Matter window : W:80 L:40
# Blood/subdural window: W:130-300 L:50-100
# Soft tissue window: W:350–400 L:20–60
# Bone window: W:2800 L:600
# Grey-white differentiation window: W:8 L:32 or W:40 L:40
BRAIN_MATTER_WINDOW = (40, 80)
SUBDURAL_WINDOW = (80, 200)
SOFT_TISSUE_WINDOW = (40, 380)
BONE_WINDOW = (600, 2800)
GRAY_WHITE_DIFFERENTIATION_WINDOW = (40, 40)

ALL_WINDOW_VALUES = [BRAIN_MATTER_WINDOW,
                     SUBDURAL_WINDOW,
                     SOFT_TISSUE_WINDOW,
                     BONE_WINDOW,
                     GRAY_WHITE_DIFFERENTIATION_WINDOW
                     ]


def image_windowed(image, custom_center=50, custom_width=130, out_side_val=False):
    '''
    Important thing to note in this function: The image migth be changed in place!
    '''
    min_value = custom_center - (custom_width / 2)
    max_value = custom_center + (custom_width / 2)

    # Including another value for values way outside the range, to (hopefully) make segmentation processes easier.
    out_value_min = custom_center - custom_width
    out_value_max = custom_center + custom_width

    if out_side_val:
        image[np.logical_and(image < min_value, image > out_value_min)] = min_value
        image[np.logical_and(image > max_value, image < out_value_max)] = max_value
        image[image < out_value_min] = out_value_min
        image[image > out_value_max] = out_value_max

    else:
        image[image < min_value] = min_value
        image[image > max_value] = max_value

    return image


def image_resample(image, dicom_header, new_spacing=[1, 1]):
    spacing = map(float, dicom_header.PixelSpacing)
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image


if __name__ == '__main__':
    # img = pydicom.read_file('ID_00019828f.dcm').pixel_array
    img = image_windowed(apply_hounsfield_transformation('ID_00019828f.dcm'), 40, 80, True)
    plt.imshow(img, cmap='bone')
    plt.show()
