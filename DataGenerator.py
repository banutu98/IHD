from keras.utils import Sequence
from Preprocessor import Preprocessor
import numpy as np
from utilities.defines import TRAIN_DIR


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
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indices]

        return self.__data_generation(list_ids_temp)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_ids_temp):
        x = np.empty((self.batch_size, *self.img_size))
        if self.labels is not None:  # training phase
            y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)
            for i, ID in enumerate(list_ids_temp):
                image = Preprocessor.preprocess(self.img_dir + ID + ".dcm")
                image = np.repeat(image[..., np.newaxis], 3, -1)
                x[i, ] = image
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
