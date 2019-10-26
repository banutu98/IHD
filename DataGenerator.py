from keras.utils import Sequence
from Preprocessor import Preprocessor
import numpy as np
import pandas as pd
import random
from utilities.defines import TRAIN_DIR
from utilities.augmentations import blur_image, noisy, adjust_brightness


class DataGenerator(Sequence):

    def __init__(self, list_ids, labels=None, batch_size=1, img_size=(512, 512, 3),
                 img_dir=TRAIN_DIR, shuffle=True, n_classes=2):
        self.list_ids = list_ids
        self.indices = np.arange(len(self.list_ids))
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.shuffle = shuffle
        self.n_classes = n_classes
        # TODO: this could be generalized with the help of
        # an Augmenter class
        self.n_augment = 3      # 3 data augmentation functions
        self.augment_funcs = [blur_image,
                              noisy,
                              adjust_brightness,
                              lambda img: img]      # identity function
        self.on_epoch_end()
        if labels is not None:
            # Weights should be a probability distribution.
            # If the number of training instances is too large,
            # there could be issues! (arithmetic underflow)
            weight_func = lambda row: 1.0 if row["any"] == 0 else self.n_augment + 1
            self.weights = labels.apply(weight_func, axis=1)
            total = self.weights.sum()
            self.weights = (self.weights / total).values

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        indices = np.random.choice(self.indices, size=self.batch_size,
                         replace=False, p=self.weights)
        #indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indices]

        # TODO: change list_ids_temp to list of indices chosen by
        # __getitem__
        return self.__data_generation(list_ids_temp)

    # Don't think this is necessary anymore, indices are sampled randomly.
    def on_epoch_end(self):
        #self.indices = np.arange(len(self.list_ids))
        #if self.shuffle:
            #np.random.shuffle(self.indices)
        pass

    def __data_generation(self, list_ids_temp):
        x = np.empty((self.batch_size, *self.img_size))
        if self.labels is not None:  # training phase
            y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)
            for i, ID in enumerate(list_ids_temp):
                # TODO: change ID to self.list_ids[ID]
                image = Preprocessor.preprocess(self.img_dir + ID + ".dcm")
                if self.labels.iloc[i]['any'] == 1:
                    # TODO: random module is NOT thread-safe, must
                    # come up later with a better solution
                    image = self.augment_funcs[random.randint(0, self.n_augment)](image)
                image = np.repeat(image[..., np.newaxis], 3, -1)
                x[i, ] = image
                # TODO: use ID as index for labels
                if self.n_classes == 2:
                    y[i, ] = self.labels.iloc[i]['any']
                else:
                    y[i, ] = self.labels.iloc[i, 1:]
            return x, y
        else:  # test phase
            for i, ID in enumerate(list_ids_temp):
                image = Preprocessor.preprocess(self.img_dir + ID + ".dcm")
                image = np.repeat(image[..., np.newaxis], 3, -1)
                x[i, ] = image
            return x
