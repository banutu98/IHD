import os
from keras.utils import Sequence
import numpy as np
from Preprocessor import Preprocessor
from utilities.utils import get_sequence_clipping_order


class LSTMDataGenerator(Sequence):

    def __init__(self, list_ids, labels=None, batch_size=1, img_size=(512, 512, 3),
                 sequence_size=20, img_dir='data/train', shuffle=True):
        # here, list_ids is a series of lists; each list represents an
        # ordered sequence of scans that compose a single study
        self.list_ids = list_ids
        self.indices = np.arange(len(self.list_ids))
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.sequence_size = sequence_size
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

    def __data_generation(self, list_ids_temp):
        x = np.empty((self.batch_size, self.sequence_size, *self.img_size))
        hounsfield_func = lambda im: Preprocessor.preprocess(os.path.join(self.img_dir, im + ".dcm"))
        if self.labels is not None:  # training phase
            y = np.empty((self.batch_size, 5), dtype=np.float32)
            for i, seq in enumerate(list_ids_temp):
                imgs = np.array(list(map(hounsfield_func, seq)))
                imgs = np.repeat(imgs[..., np.newaxis], 3, -1)
                diff = len(seq) - self.sequence_size
                if diff < 0:
                    padding = np.repeat(np.zeros(imgs.shape[1:])[np.newaxis, ...], abs(diff), 0)
                    imgs = np.concatenate((imgs, padding), axis=0)
                elif diff > 0:
                    indices = get_sequence_clipping_order(len(seq))
                    imgs = np.delete(imgs, indices[:diff], 0)
                x[i, ] = imgs
                y[i, ] = self.labels.iloc[i, 1:]
            return x, y
        else:                       # test phase
            for i, seq in enumerate(list_ids_temp):
                imgs = np.array(list(map(hounsfield_func, seq)))
                imgs = np.repeat(imgs[..., np.newaxis], 3, -1)
                diff = len(seq) - self.sequence_size
                if diff < 0:
                    padding = np.repeat(np.zeros(imgs.shape[1:])[np.newaxis, ...], abs(diff), 0)
                    imgs = np.concatenate((imgs, padding), axis=0)
                elif diff > 0:
                    indices = get_sequence_clipping_order(len(seq))
                    imgs = np.delete(imgs, indices[:diff], 0)
                x[i, ] = imgs
            return x
