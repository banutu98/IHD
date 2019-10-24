import glob
import os
import sys

import keras
import numpy as np
import pandas as pd
from keras.optimizers import Adamax

from DataGenerator import DataGenerator
from LSTMDataGenerator import LSTMDataGenerator
from NeuralNetwork import StandardModel
from utilities.defines import TRAIN_DIR, MODELS_DIR
from utilities.utils import get_study_sequences
from utilities.utils import print_error
from sklearn.metrics import log_loss


def prepare_data():
    csv = pd.read_csv(os.path.join(TRAIN_DIR, 'labels.csv'))
    files = glob.glob(os.path.join(TRAIN_DIR, "*.dcm"))
    files = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], files))
    filtered_csv = csv[csv.id.isin(files)]
    indices = np.random.rand(len(filtered_csv))
    mask = indices < 0.2
    x_train, y_train = filtered_csv[mask].id, filtered_csv.iloc[mask, 1:]
    x_test, y_test = filtered_csv[~mask].id, filtered_csv.iloc[~mask, 1:]
    x_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    x_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    return x_train, y_train, x_test, y_test


def extract_labels(csv, sequences):
    result_labels = list()
    for seq in sequences:
        current_labels = csv[csv.id.isin(seq)]
        current_labels = current_labels.iloc[:, 1:]
        current_labels.reset_index(inplace=True, drop=True)
        current_labels = current_labels.iloc[:, 1:]
        result_labels.append(current_labels)
    return result_labels


def prepare_sequential_data():
    csv = pd.read_csv(os.path.join(TRAIN_DIR, 'labels.csv'))
    sequences = get_study_sequences()
    indices = np.random.rand(len(sequences))
    mask = indices < 0.1
    x_train, x_test = sequences.iloc[mask], sequences.iloc[~mask]
    y_train, y_test = extract_labels(csv, x_train), extract_labels(csv, x_test)
    x_train.reset_index(inplace=True, drop=True)
    x_test.reset_index(inplace=True, drop=True)
    return x_train, y_train, x_test, y_test


def train_binary_model(base_model, already_trained_model=None):
    x_train, y_train, x_test, y_test = prepare_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=2, use_softmax=True)
        model = model.build_model()
        model.compile(Adamax(), loss='categorical_crossentropy', metrics=['acc'])
        model.fit_generator(DataGenerator(x_train, labels=y_train, n_classes=2))
        model.save(os.path.join(MODELS_DIR, 'model.h5'))
        loss, accuracy = model.evaluate_generator(DataGenerator(x_test, labels=y_test, n_classes=2))
        print(loss, accuracy)
    else:
        model_path = os.path.join(MODELS_DIR, already_trained_model)
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            loss, accuracy = model.evaluate_generator(DataGenerator(x_test, labels=y_test, n_classes=2))
            print(loss, accuracy)
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def train_multi_class_model(base_model, already_trained_model=None):
    x_train, y_train, x_test, y_test = prepare_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=5, use_softmax=True)
        model = model.build_model()
        model.compile(Adamax(), loss='categorical_crossentropy', metrics=['acc'])
        model.fit_generator(DataGenerator(x_train, labels=y_train, n_classes=5))
        model.save(os.path.join(MODELS_DIR, 'model.h5'))
        y_pred = model.predict_generator(DataGenerator(x_test, n_classes=5))
        y_test = y_test.iloc[:, 1:]
        print(log_loss(y_test, y_pred))
    else:
        model_path = os.path.join(MODELS_DIR, already_trained_model)
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            y_pred = model.predict_generator(DataGenerator(x_test, n_classes=5))
            y_test = y_test.iloc[:, 1:]
            print(log_loss(y_test, y_pred))
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


# TODO: RecurrentModel Fusion + Sequences Same Size
def train_recurrent_multi_class_model(base_model, already_trained_model=None):
    x_train, y_train, x_test, y_test = prepare_sequential_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=5, use_softmax=False)
        model = model.build_model()
        model.compile(Adamax(), loss='categorical_crossentropy', metrics=['acc'])
        model.fit_generator(LSTMDataGenerator(x_train, labels=y_train))
        model.save(os.path.join(MODELS_DIR, 'model.h5'))
        y_pred = model.predict_generator(LSTMDataGenerator(x_test))
        print(log_loss(y_test, y_pred))
    else:
        model_path = os.path.join(MODELS_DIR, already_trained_model)
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            y_pred = model.predict_generator(LSTMDataGenerator(x_test))
            print(log_loss(y_test, y_pred))
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def main():
    # TODO: Possible MODELS for training: inception, xception, resnet, densenet, nas
    # train_binary_model('xception')
    # train_multi_class_model('densenet')
    prepare_sequential_data()
    train_recurrent_multi_class_model('xception')


if __name__ == '__main__':
    main()
