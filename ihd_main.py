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
from utilities.defines import TRAIN_DIR
from utilities.utils import print_error
from sklearn.metrics import log_loss
from Preprocessor import Preprocessor


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


def prepare_sequential_data():
    # open label + metadata CSV
    csv = pd.read_csv(os.path.join(TRAIN_DIR, "train_meta.csv"))
    # sort by study ID and position
    csv.sort_values(by=["StudyInstanceUID", "ImagePositionPatient3"], inplace=True, ascending=False)
    label_columns = ["any", "epidural", "intraparenchymal",
                     "intraventricular", "subarachnoid", "subdural"]
    # filter unnecessary columns
    csv = csv[["StudyInstanceUID", "id"] + label_columns]
    # get sequences of IDs (groupby preserves order)
    sequences = csv.groupby("StudyInstanceUID")["id"].apply(list)
    # group labels into one single column
    csv["labels"] = csv[label_columns].values.tolist()
    # get sequences of labels
    labels = csv.groupby("StudyInstanceUID")["labels"].apply(list)
    indices = np.random.rand(sequences.size)
    # partition data
    mask = indices < 0.1
    x_train, x_test = list(sequences.iloc[mask]), list(sequences.iloc[~mask])
    y_train, y_test = list(labels.iloc[mask]), list(labels.iloc[~mask])
    return x_train, y_train, x_test, y_test


def train_binary_model(base_model, already_trained_model=None):
    x_train, y_train, x_test, y_test = prepare_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=2, use_softmax=True)
        model = model.build_model()
        model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
        model.fit_generator(DataGenerator(x_train, labels=y_train, n_classes=2))
        model.save('model.h5')
        loss, accuracy = model.evaluate_generator(DataGenerator(x_test, labels=y_test, n_classes=2))
        print(loss, accuracy)
    else:
        if os.path.exists(already_trained_model):
            model = keras.models.load_model(already_trained_model)
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
        model.save('model.h5')
        y_pred = model.predict_generator(DataGenerator(x_test, n_classes=5))
        y_test = y_test.iloc[:, 1:]
        print(log_loss(y_test, y_pred))
    else:
        if os.path.exists(already_trained_model):
            model = keras.models.load_model(already_trained_model)
            y_pred = model.predict_generator(DataGenerator(x_test, n_classes=5))
            y_test = y_test.iloc[:, 1:]
            print(log_loss(y_test, y_pred))
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def train_recurrent_multi_class_model(base_model, already_trained_model=None):
    x_train, y_train, x_test, y_test = prepare_sequential_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=5, use_softmax=False, pooling_method=None)
        model = model.build_model()
        model.compile(Adamax(), loss='categorical_crossentropy', metrics=['acc'])
        model.fit_generator(LSTMDataGenerator(x_train, labels=y_train))
        model.save('model.h5')
        y_pred = model.predict_generator(LSTMDataGenerator(x_test))
        print(log_loss(y_test, y_pred))
    else:
        if os.path.exists(already_trained_model):
            model = keras.models.load_model(already_trained_model)
            y_pred = model.predict_generator(LSTMDataGenerator(x_test))
            print(log_loss(y_test, y_pred))
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def test_recurrent_network():
    def generate_single_instance(instance):
        images, labels = list(), list()
        for file in instance:
            file_path = os.path.join(TRAIN_DIR, file)
            images.append(Preprocessor.preprocess(file_path))
            labels.append(np.random.uniform(0, 1, 5))
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)
        return images, labels

    model = StandardModel('xception', (512, 512, 3), classes=5, use_softmax=False, pooling_method=None)
    model = model.build_model()
    model.compile(Adamax(), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    keras.utils.plot_model(model, show_shapes=True)
    x_train = []
    y_train = []
    data = [['ID_00025ef4b.dcm', 'ID_00027c277.dcm', 'ID_00027cbb1.dcm'],
            ['ID_000229f2a.dcm', 'ID_000230ed7.dcm', 'ID_000270f8b.dcm'],
            ['ID_00025ef4b.dcm', 'ID_00027c277.dcm', 'ID_00027cbb1.dcm']]
    for i in range(3):
        instance_images, instance_labels = generate_single_instance(data[i])
        x_train.append(instance_images)
        y_train.append(instance_labels)
    x_train = np.stack(x_train)
    x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
    y_train = np.stack(y_train)
    print(x_train.shape, y_train.shape)
    model.fit(x_train, y_train, batch_size=1)


def main():
    # TODO: Possible MODELS for training: inception, xception, resnet, densenet, nas
    # train_binary_model('xception')
    # train_multi_class_model('densenet')
    # prepare_sequential_data()
    train_recurrent_multi_class_model('xception')
    # test_recurrent_network()


if __name__ == '__main__':
    main()
