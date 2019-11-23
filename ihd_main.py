import glob
import sys

import keras
import pandas as pd
from keras.optimizers import Adamax

from utilities.Output import create_output_csv
from NeuralNetwork import *
from generators.DataGenerator import *
from generators.LSTMDataGenerator import *


def prepare_data(only_positives=False):
    csv = pd.read_csv(os.path.join('data/train', 'labels_2.csv'))
    files = glob.glob(os.path.join(TRAIN_DIR_STAGE_2, "*.dcm"))
    files = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], files))
    filtered_csv = csv[csv.id.isin(files)]
    if only_positives:
        filtered_csv = filtered_csv.loc[filtered_csv['any'] == 1]
    indices = np.random.rand(len(filtered_csv))
    mask = indices < 0.9
    x_train, y_train = list(filtered_csv[mask].id), filtered_csv.iloc[mask, 1:]
    x_test, y_test = list(filtered_csv[~mask].id), filtered_csv.iloc[~mask, 1:]
    # x_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    # x_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    return x_train, y_train, x_test, y_test


def prepare_sequential_data(only_positives=False, for_prediction=False):
    if not for_prediction:
        # open label + metadata CSV
        csv = pd.read_csv(os.path.join('data/train', "train_meta_2.csv"))
        # sort by study ID and position
        csv.sort_values(by=["StudyInstanceUID", "ImagePositionPatient3"], inplace=True, ascending=False)
        label_columns = ["any", "epidural", "intraparenchymal",
                         "intraventricular", "subarachnoid", "subdural"]
        # filter unnecessary columns
        csv = csv[["StudyInstanceUID", "id"] + label_columns]
        if only_positives:
            csv = csv.loc[csv['any'] == 1]
        # get sequences of IDs (groupby preserves order)
        sequences = csv.groupby("StudyInstanceUID")["id"].apply(list)
        # group labels into one single column
        csv["labels"] = csv[label_columns].values.tolist()
        # get sequences of labels
        labels = csv.groupby("StudyInstanceUID")["labels"].apply(list)
#         indices = np.random.rand(sequences.size)
#         # partition data
#         mask = indices < 0.001
#         x_train = list(sequences.iloc[mask])
#         y_train = list(labels.iloc[mask])
        x_train = list(sequences)
        y_train = list(labels)
        return x_train, y_train
    else:
        csv = pd.read_csv(os.path.join('data/train', "test_meta_2.csv"))
        # sort by study ID and position
        csv.sort_values(by=["StudyInstanceUID", "ImagePositionPatient3"], inplace=True, ascending=False)
        # filter unnecessary columns
        csv = csv[["StudyInstanceUID", "id"]]
        # get sequences of IDs (groupby preserves order)
        sequences = csv.groupby("StudyInstanceUID")["id"].apply(list)
        x_test = list(sequences)
        return x_test


def train_binary_model(base_model, model_name, already_trained_model=None):
    x_train, y_train, x_test, y_test = prepare_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=2, use_softmax=True)
        model = model.build_model()
        model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
        model.fit_generator(DataGenerator(x_train, labels=y_train, n_classes=2, batch_size=8), epochs=1)
        model.save(model_name)
    else:
        if os.path.exists(already_trained_model):
            model = keras.models.load_model(already_trained_model)
            model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
            model.fit_generator(DataGenerator(x_train, labels=y_train, n_classes=2, batch_size=8), epochs=1)
            model.save(model_name)
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def train_multi_class_model(base_model, model_name, already_trained_model=None, n_classes=5):
    x_train, y_train, x_test, y_test = prepare_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=n_classes, use_softmax=True)
        model = model.build_model()
        model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
        model.fit_generator(DataGenerator(x_train, labels=y_train, n_classes=n_classes, batch_size=8), epochs=3)
        model.save(model_name)
    else:
        if os.path.exists(already_trained_model):
            model = keras.models.load_model(already_trained_model)
            model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
            model.fit_generator(DataGenerator(x_train, labels=y_train, n_classes=n_classes, batch_size=8), epochs=3)
            model.save(model_name)
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def train_recurrent_multi_class_model(base_model, model_name, already_trained_model=None):
    x_train, y_train = prepare_sequential_data()
    if not already_trained_model:
        model = StandardModel(base_model, (512, 512, 3), classes=5, use_softmax=False, pooling_method=None)
        model = model.build_model()
        model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
        model.fit_generator(LSTMDataGenerator(x_train, labels=y_train), epochs=3)
        model.save(model_name)
    else:
        if os.path.exists(already_trained_model):
            model = keras.models.load_model(already_trained_model)
            model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
            model.fit_generator(LSTMDataGenerator(x_train, labels=y_train), epochs=3)
            model.save(model_name)
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def construct_probabilities_sequences(x_train, y_train, loaded_multi_class_model):
    def preprocess_func(im):
        return Preprocessor.preprocess(os.path.join(TRAIN_DIR_STAGE_2, im + ".dcm"))
    new_x_train = list()
    new_y_train = list()
    print(len(x_train))
    count = 1
    ideal_length = max([len(seq) for seq in x_train])
    for seq, label_seq in zip(x_train, y_train):
        print(count)
        label_seq = np.array(label_seq)
        padding = np.zeros((ideal_length, 6))
        label_padding = np.zeros((ideal_length, 6))
        preprocessed_seq = np.array(list(map(preprocess_func, seq)))
        preprocessed_seq = np.array([np.repeat(p[..., np.newaxis], 3, -1) for p in preprocessed_seq])
        predictions = loaded_multi_class_model.predict(preprocessed_seq)
        padding[:predictions.shape[0], :predictions.shape[1]] = predictions
        padding = padding.reshape(1, *padding.shape)
        label_padding[:label_seq.shape[0], :label_seq.shape[1]] = label_seq
        label_padding = label_padding.reshape(1, *label_padding.shape)
        new_x_train.append(padding)
        new_y_train.append(label_padding)
        count += 1
    new_x_train = np.concatenate(new_x_train, axis=0)
    new_y_train = np.concatenate(new_y_train, axis=0)
    return new_x_train, new_y_train


def train_simple_recurrent_model(multi_class_model, model_name, already_trained_model=None):
    x_train, y_train = prepare_sequential_data()
    loaded_multi_class_model = keras.models.load_model(multi_class_model)
    x_train, y_train = construct_probabilities_sequences(x_train, y_train, loaded_multi_class_model)
    print(x_train.shape, y_train.shape)
    if not already_trained_model:
        model = StandardModel(classes=6)
        model = model.build_simple_recurrent_model()
        model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
        model.fit(x_train, y_train, epochs=5, batch_size=1)
        model.save(model_name)
    else:
        if os.path.exists(already_trained_model):
            model = keras.models.load_model(already_trained_model)
            model.compile(Adamax(), loss='binary_crossentropy', metrics=['acc'])
            model.fit(x_train, y_train, epochs=5, batch_size=1)
            model.save(model_name)
        else:
            print_error("Provided model file doesn't exist! Exiting...")
            sys.exit(1)


def predict_multiclass_all(multi_class_model):
    loaded_multi_class_model = keras.models.load_model(multi_class_model)
    output_dict = dict()
    index = 1
    for filename in os.scandir(TEST_DIR_STAGE_2):
        preprocessed_image = Preprocessor.preprocess(filename.path)
        preprocessed_image = np.repeat(preprocessed_image[..., np.newaxis], 3, -1)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        classes_predictions = loaded_multi_class_model.predict(preprocessed_image)[0]
        file_id = os.path.splitext(filename.name)[0]
        output_dict[file_id] = tuple(classes_predictions)
        index += 1
        if index % 1000 == 0:
            print(index)
    create_output_csv(output_dict)


def predict(binary_model, multi_class_model, conditional_probabilities=True):
    loaded_binary_model = keras.models.load_model(binary_model)
    loaded_multi_class_model = keras.models.load_model(multi_class_model)

    output_dict = dict()
    index = 1
    for filename in os.scandir(TEST_DIR_STAGE_2):
        preprocessed_image = Preprocessor.preprocess(filename.path)
        preprocessed_image = np.repeat(preprocessed_image[..., np.newaxis], 3, -1)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        binary_prediction = loaded_binary_model.predict(preprocessed_image)
        file_id = os.path.splitext(filename.name)[0]
        any_prediction = binary_prediction[0][0]
        classes_predictions = loaded_multi_class_model.predict(preprocessed_image)[0]
        if conditional_probabilities:
            classes_predictions = classes_predictions * any_prediction
            output_dict[file_id] = tuple(np.concatenate((np.array([any_prediction]),
                                                         classes_predictions), axis=None))
        else:
            if any_prediction >= 0.5:
                output_dict[file_id] = tuple(np.concatenate((np.array([1]),
                                                             classes_predictions), axis=None))
            else:
                output_dict[file_id] = tuple(0, 0, 0, 0, 0, 0)
        index += 1
        if index % 1000 == 0:
            print(index)
    create_output_csv(output_dict)


def recurrent_predict(multi_class_model, recurrent_model):
    def preprocess_func(im):
        return Preprocessor.preprocess(os.path.join(TEST_DIR_STAGE_2, im + ".dcm"))
    loaded_multi_class_model = keras.models.load_model(multi_class_model)
    loaded_recurrent_model = keras.models.load_model(recurrent_model)
    x_test = prepare_sequential_data(for_prediction=True)

    output_dict = dict()
    index = 1
    print(len(x_test))
    for seq in x_test:
        preprocessed_images = list()
        preprocessed_seq = np.array(list(map(preprocess_func, seq)))
        preprocessed_seq = np.array([np.repeat(p[..., np.newaxis], 3, -1) for p in preprocessed_seq])
        predictions = loaded_multi_class_model.predict(preprocessed_seq)
        predictions = predictions.reshape(1, *predictions.shape)
        classes_predictions = loaded_recurrent_model.predict(predictions)[0]
        for i in range(len(seq)):
            current_classes_predictions = classes_predictions[i]
            output_dict[seq[i]] = tuple(current_classes_predictions)
            index += 1
            if index % 1000 == 0:
                print(index)
    create_output_csv(output_dict)


def main():
    # TODO: Possible MODELS for training: inception, xception, resnet, densenet, nas
    # train_binary_model('xception', 'binary_model_improved.h5', 'binary_model.h5')
    # predict('binary_model_improved.h5', 'categorical_model_v3_full_improved.h5')
    # prepare_sequential_data()
    # train_multi_class_model('xception', 'categorical_model_v3_full_improved.h5', 'categorical_model_v3_full.h5')
    # train_multi_class_model('xception', 'categorical_model_six_full_improved_v5.h5', 'categorical_model_six_full_improved.h5', n_classes=6)
    # predict_multiclass_all('categorical_model_six_full_improved_v5.h5')
    # test_recurrent_network()
    # train_recurrent_multi_class_model('xception', 'recurrent_model.h5')
    # extract_csv_partition()
    # extract_metadata(data_prefix=TEST_DIR_STAGE_2)
    # train_simple_recurrent_model('categorical_model_six_full_improved_v5.h5', 'recurrent_model_improved_v5.h5')
    recurrent_predict('categorical_model_six_full_improved_v5.h5', 'recurrent_model_improved_v5.h5')
    # recurrent_predict('categorical_model_six_full_improved.h5', 'recurrent_model_improved_v5.h5')


if __name__ == '__main__':
    main()
