import matplotlib.pyplot as plt
import pydicom
import os


def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if rescale:
        img = (img - img_min) / (img_max - img_min)

    return img


def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


if __name__ == '__main__':
    case = os.path.join('.', 'ID_000012eaf.dcm')

    data = pydicom.read_file(case)

    intercept, slope = get_windowing(data)
    img = pydicom.read_file(case).pixel_array

    # There are at least 5 windows that a radiologist goes through for each scan!
    # Brain Matter window : W:80 L:40
    # Blood/subdural window: W:130-300 L:50-100
    # Soft tissue window: W:350–400 L:20–60
    # Bone window: W:2800 L:600
    # Grey-white differentiation window: W:8 L:32 or W:40 L:40
    windowed_img = window_image(img, 600, 2800, intercept, slope)

    plt.imshow(windowed_img, cmap=plt.cm.bone)
    plt.show()
