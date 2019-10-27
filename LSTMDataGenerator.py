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
        return self.__data_generation(indices)

    def on_epoch_end(self):
        pass

    def __data_generation(self, indices):
        x = np.empty((self.batch_size, self.sequence_size, *self.img_size))
        preprocess_func = lambda im: Preprocessor.preprocess(os.path.join(self.img_dir, im + ".dcm"))
        if self.labels is not None:  # training phase
            y = np.empty((self.batch_size, 5), dtype=np.float32)
            for i, idx in enumerate(indices):
                seq = self.list_ids[idx]
                imgs = map(preprocess_func, seq)
                # TODO: must come up with a representation for
                # the labels of seqeunce; maybe a nested DataFrame?
                if self.labels.iloc[idx]['any'] == 1:
                    func_idxs = np.random.randint(0, self.n_augment+1,
                                                  size=len(seq))
                    imgs = [self.augment_funcs[j](img)
                            for j, img in zip(func_idxs, imgs)]
                imgs = np.array(imgs)
                imgs = np.repeat(imgs[..., np.newaxis], 3, -1)
                diff = len(seq) - self.sequence_size
                if diff < 0:
                    padding = np.repeat(np.zeros(imgs.shape[1:])[np.newaxis, ...], abs(diff), 0)
                    imgs = np.concatenate((imgs, padding), axis=0)
                elif diff > 0:
                    indices = get_sequence_clipping_order(len(seq))
                    imgs = np.delete(imgs, indices[:diff], 0)
                x[i, ] = imgs
                y[i, ] = self.labels.iloc[idx, 1:]
            return x, y
        else:                       # test phase
            for i, idx in enumerate(indices):
                seq = self.list_ids[idx]
                imgs = np.array(list(map(preprocess_func, seq)))
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
