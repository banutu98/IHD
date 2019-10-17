import pydicom
import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):

    def __init__(self, list_ids, labels=None, batch_size=1, img_size=(512, 512, 3),
                 img_dir='data/train', shuffle=True):
        self.list_ids = list_ids
        self.indices = np.arange(len(self.list_ids))
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.shuffle = shuffle
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

    @staticmethod
    def _read(path):
        return pydicom.dcmread(path).pixel_array

    def __data_generation(self, list_ids_temp):
        x = np.empty((self.batch_size, *self.img_size))
        if self.labels is not None:  # training phase
            y = np.empty((self.batch_size, 6), dtype=np.float32)
            for i, ID in enumerate(list_ids_temp):
                x[i, ] = self._read(self.img_dir + ID + ".dcm")
                y[i, ] = self.labels.loc[ID].values
            return x, y
        else:  # test phase
            for i, ID in enumerate(list_ids_temp):
                x[i, ] = self._read(self.img_dir + ID + ".dcm")
            return x
