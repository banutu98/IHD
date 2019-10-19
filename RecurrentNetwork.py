import sys

from keras.layers import Dense, Dropout, BatchNormalization, LSTM, ConvLSTM2D, AveragePooling2D
from keras.models import Model

from utilities.utils import print_error


class RecurrentModel:

    def __init__(self, sequence_length, convolutions_shape=None, use_probabilities=True):
        self.sequence_length = sequence_length
        self.use_probabilities = use_probabilities
        self.convolutions_shape = convolutions_shape

    def build_model(self):
        return self.build_probs_model() if self.use_probabilities else self.build_convs_model()

    def build_probs_model(self):
        input_lstm = LSTM(256, dropout=0.3, recurrent_dropout=0.3, activation='softsign',
                          input_shape=(5, self.sequence_length))  # 5 probabilities and ~25 sequence_length
        relu_layer = Dense(128, activation='relu')(input_lstm)
        dropout_layer = Dropout(0.2)(relu_layer)
        final_relu_layer = Dense(64, activation='relu')(dropout_layer)
        output_layer = Dense(5, activation='softmax')(final_relu_layer)
        return Model(inputs=input_lstm, outputs=output_layer)

    def build_convs_model(self):
        if self.convolutions_shape is None:
            print_error('Convolutions shape is missing! Aborting!')
            sys.exit(1)
        input_shape = tuple([s for s in self.convolutions_shape] + [self.sequence_length])
        input_lstm = ConvLSTM2D(64, (3, 3), padding='same', activation='softsign', dropout=0.2, recurrent_dropout=0.2,
                                input_shape=input_shape)
        batch_norm = BatchNormalization()(input_lstm)
        average_pool = AveragePooling2D()(batch_norm)
        dense_layer = Dense(64, activation='relu')(average_pool)
        output_layer = Dense(5, activation='softmax')(dense_layer)
        return Model(inputs=input_lstm, outputs=output_layer)
