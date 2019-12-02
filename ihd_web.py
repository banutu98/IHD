import keras
from generators.DataGenerator import *

def get_model(name):
    if name == "single":
        model_path = os.path.join('..', 'models', 'categorical_model_six_full_improved_v5.h5')
        return keras.models.load_model(model_path, compile=False)
    elif name == "sequential":
        multi_model_path = os.path.join('..', 'models', 'categorical_model_six_full_improved_v5.h5')
        recurrent_model_path = os.path.join('..', 'models', 'recurrent_model_improved_v5.h5')
        multi_class_model = keras.models.load_model(multi_model_path, compile=False)
        recurrent_model = keras.models.load_model(recurrent_model_path, compile=False)
        return multi_class_model, recurrent_model
    else:
        raise ValueError("Name %s not recognized." % (name,))


def predict_single_file(file):
    loaded_multi_class_model = get_model("single")
    preprocessed_image = Preprocessor.preprocess(file)
    preprocessed_image = np.repeat(preprocessed_image[..., np.newaxis], 3, -1)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    classes_predictions = loaded_multi_class_model.predict(preprocessed_image)[0]
    return classes_predictions


def predict_file_sequence(files):
    def preprocess_func(file_path):
        return Preprocessor.preprocess(file_path)
    loaded_multi_class_model, loaded_recurrent_model = get_model("sequential")
    preprocessed_files = np.array(list(map(preprocess_func, files)))
    preprocessed_files = np.array([np.repeat(p[..., np.newaxis], 3, -1) for p in preprocessed_files])
    predictions = loaded_multi_class_model.predict(preprocessed_files)
    predictions = predictions.reshape(1, *predictions.shape)
    classes_predictions = loaded_recurrent_model.predict(predictions)[0]
    return classes_predictions
