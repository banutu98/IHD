import keras
from generators.DataGenerator import *


def predict_single_file(file):
    model_path = os.path.join('..', 'models', 'categorical_model_six_full_improved_v5.h5')
    loaded_multi_class_model = keras.models.load_model(model_path)
    preprocessed_image = Preprocessor.preprocess(file)
    preprocessed_image = np.repeat(preprocessed_image[..., np.newaxis], 3, -1)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    classes_predictions = loaded_multi_class_model.predict(preprocessed_image)[0]
    return classes_predictions


def predict_file_sequence(files):
    def preprocess_func(file_path):
        return Preprocessor.preprocess(file_path)
    multi_model_path = os.path.join('..', 'models', 'categorical_model_six_full_improved_v5.h5')
    recurrent_model_path = os.path.join('..', 'models', 'recurrent_model_improved_v5.h5')
    loaded_multi_class_model = keras.models.load_model(multi_model_path)
    loaded_recurrent_model = keras.models.load_model(recurrent_model_path)

    preprocessed_files = np.array(list(map(preprocess_func, files)))
    preprocessed_files = np.array([np.repeat(p[..., np.newaxis], 3, -1) for p in preprocessed_files])
    predictions = loaded_multi_class_model.predict(preprocessed_files)
    predictions = predictions.reshape(1, *predictions.shape)
    classes_predictions = loaded_recurrent_model.predict(predictions)[0]
    return classes_predictions
