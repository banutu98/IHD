from keras.applications import NASNetLarge, InceptionResNetV2, Xception, DenseNet201, ResNet50
from keras.layers import Bidirectional, LSTM, TimeDistributed, Masking
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input, GlobalAveragePooling2D
from keras.models import Sequential, Model

from utilities.utils import print_error


class StandardModel:

    def __init__(self, network, input_shape, pooling_method='max', classes=2, use_softmax=True):
        self.base_model = self.get_base_model(network, input_shape, pooling_method)
        self.classes = classes
        self.use_softmax = use_softmax
        self.input_shape = input_shape

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
        model.add(Dense(2, activation='sigmoid'))
        return model

    def build_multi_class_model(self):
        return self.build_probability_model() if self.use_softmax else self.build_recurrent_model()

    def build_probability_model(self):
        model = Sequential()
        model.add(self.base_model)
        model.add(Dense(96))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.3))
        model.add(Dense(5, activation='softmax'))
        return model

    def build_recurrent_model(self):
        inputs = Input(shape=(40, *self.input_shape))
        time_dist = TimeDistributed(self.base_model)(inputs)
        global_pool = TimeDistributed(GlobalAveragePooling2D())(time_dist)
        dense_relu = TimeDistributed(Dense(256, activation='relu'))(global_pool)

        masked = Masking(0.0)(dense_relu)
        out = Bidirectional(LSTM(256, return_sequences=True, activation='softsign',
                                 dropout=0.2, recurrent_dropout=0.2))(masked)
        out = TimeDistributed(Dense(5, activation='softmax'))(out)

        model = Model(inputs=inputs, outputs=out)
        return model
