import matplotlib.pyplot as plt
import scipy.ndimage
import pydicom
import numpy as np
from skimage import morphology

from Window import image_resample
from Window import image_windowed
from Hounsfield import apply_hounsfield_transformation


def image_background_segmentation(image_path, WL=40, WW=80, display=False):
    # img = image_to_hu(image_path)
    img = apply_hounsfield_transformation(image_path)
    dcm_head = pydicom.read_file(image_path)
    img = image_resample(img, dcm_head)
    img_out = img.copy()
    # use values outside the window as well, helps with segmentation
    img = image_windowed(img, custom_center=WL, custom_width=WW, out_side_val=True)

    # Calculate the outside values by hand (again)
    lB = WW - WL
    uB = WW + WL

    # Keep only values inside of the window
    background_separation = np.logical_and(img > lB, img < uB)

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

    if display:
        show_images(background_separation, image_path, img, mask)

    return mask * img_out


def show_images(background_separation, image_path, img, mask):
    plt.figure(figsize=(15, 2.5))
    plt.subplot(141)
    plt.imshow(img, cmap='bone')
    plt.title('Original Images')
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(background_separation)
    plt.title('Segmentation')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(mask)
    plt.title('Mask')
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(mask * img, cmap='bone')
    plt.title('Image * Mask')
    plt.suptitle(image_path)
    plt.axis('off')


if __name__ == '__main__':
    image_background_segmentation('ID_00019828f.dcm', 40, 80, display=True)
    plt.show()
