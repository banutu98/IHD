from keras.applications import NASNetLarge, InceptionResNetV2, Xception, DenseNet201, ResNet50
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, AveragePooling2D, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, Adamax, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils import plot_model
from keras.utils import to_categorical
from utilities.utils import print_error


class MyModel:

    def __init__(self, network, input_shape, pooling_method='max', classes=2):
        self.base_model = self.get_base_model(network, input_shape, pooling_method)
        self.classes = classes

    @staticmethod
    def get_base_model(network, input_shape, pooling_method):
        if network.lower() == 'nas':
            if input_shape != (331, 331, 3):
                print_error('WARNING! The input shape is not the default one!!! Proceeding anyway!')
            return NASNetLarge(input_shape=input_shape, include_top=False, pooling=pooling_method,
                               weights=None)
        elif network.lower() == 'inception':
            if input_shape != (299, 299, 3):
                print_error('WARNING! The input shape is not the default one!!! Proceeding anyway!')
            return InceptionResNetV2(input_shape=input_shape, include_top=False, pooling=pooling_method,
                                     weights='imagenet')
        elif network.lower() == 'xception':
            if input_shape != (299, 299, 3):
                print_error('WARNING! The input shape is not the default one!!! Proceeding anyway!')
            return Xception(input_shape=input_shape, include_top=False, pooling=pooling_method,
                            weights='imagenet')
        elif network.lower() == 'densenet':
            if input_shape != (224, 224, 3):
                print_error('WARNING! The input shape is not the default one!!! Proceeding anyway!')
            return DenseNet201(input_shape=input_shape, include_top=False, pooling=pooling_method,
                               weights='imagenet')
        elif network.lower() == 'resnet':
            if input_shape != (224, 224, 3):
                print_error('WARNING! The input shape is not the default one!!! Proceeding anyway!')
            return ResNet50(input_shape=input_shape, include_top=False, pooling=pooling_method,
                            weights='imagenet')
        else:
            print_error(f'Invalid network name: {network}! Please choose from: \n ')
            return None

    def build_model(self):
        return self.build_binary_model() if self.classes == 2 else self.build_multi_class_model()

    def build_binary_model(self):
        return self.base_model

    def build_multi_class_model(self):
        return self.base_model
