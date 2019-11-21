from enum import Enum
import os

TRAIN_DIR = os.path.join('../stage_1_train_images', '')
TRAIN_DIR_STAGE_2 = os.path.join('../stage_2_train_images', '')
TEST_DIR = os.path.join('../stage_1_test_images', '')
TEST_DIR_STAGE_2 = os.path.join('../stage_2_test_images', '')
CSV_FILENAME = 'submission.csv'


class HemorrhageTypes(Enum):
    ANY = "any"
    EP = "epidural"
    IN_PA = "intraparenchymal"
    IN_VE = "intraventricular"
    SUB_AR = "subarachnoid"
    SUB_DU = "subdural"


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

ALL_WINDOW_VALUES = {'BRAIN_MATTER': BRAIN_MATTER_WINDOW,
                     'SUBDURAL': SUBDURAL_WINDOW,
                     'SOFT_TISSUE': SOFT_TISSUE_WINDOW,
                     'BONE': BONE_WINDOW,
                     'GRAY_WHITE': GRAY_WHITE_DIFFERENTIATION_WINDOW}

KERNEL_WIDTH = 13
KERNEL_HEIGHT = 13
GAUSS_MEAN = 0.1
GAUSS_STDDEV = 0.05
BRIGHTNESS_DELTA = 0.4