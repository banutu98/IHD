from keras.applications import NASNetLarge, InceptionResNetV2, Xception, DenseNet201, ResNet50
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential

from utilities.utils import print_error


class StandardModel:

    def __init__(self, network, input_shape, pooling_method='max', classes=2, use_softmax=True):
        self.base_model = self.get_base_model(network, input_shape, pooling_method)
        self.classes = classes
        self.use_softmax = use_softmax

    @staticmethod
    def get_base_model(network, input_shape, pooling_method):
        network = network.lower()
        input_warning_message = 'WARNING! The input shape is not the default one!!! Proceeding anyway!'
        if network == 'nas':
            if input_shape != (331, 331, 3):
                print_error(input_warning_message)
            return NASNetLarge(input_shape=input_shape, include_top=False, pooling=pooling_method,
                               weights=None)
        elif network == 'inception':
            if input_shape != (299, 299, 3):
                print_error(input_warning_message)
            return InceptionResNetV2(input_shape=input_shape, include_top=False, pooling=pooling_method,
                                     weights='imagenet')
        elif network == 'xception':
            if input_shape != (299, 299, 3):
                print_error(input_warning_message)
            return Xception(input_shape=input_shape, include_top=False, pooling=pooling_method,
                            weights='imagenet')
        elif network == 'densenet':
            if input_shape != (224, 224, 3):
                print_error(input_warning_message)
            return DenseNet201(input_shape=input_shape, include_top=False, pooling=pooling_method,
                               weights='imagenet')
        elif network == 'resnet':
            if input_shape != (224, 224, 3):
                print_error(input_warning_message)
            return ResNet50(input_shape=input_shape, include_top=False, pooling=pooling_method,
                            weights='imagenet')
        else:
            print_error(f'Invalid network name: {network}! Please choose from: \n ')
            return None

    def build_model(self):
        return self.build_binary_model() if self.classes == 2 else self.build_multi_class_model()

    def build_binary_model(self):
        model = Sequential()
        model.add(self.base_model)
        model.add(Dense(96))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        return model

    def build_multi_class_model(self):
        model = Sequential()
        model.add(self.base_model)
        if self.use_softmax:
            model.add(Dense(96))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.3))
            model.add(Dense(5, activation='softmax'))
        return model
