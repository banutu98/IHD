import os
import unittest
import numpy as np
from Preprocessor import Preprocessor
import keras
from NeuralNetwork import StandardModel


class TestNeuralNetwork(unittest.TestCase):

    def _init_test_images(self):
        real_test_image = Preprocessor.preprocess('data/ID_000178e76.dcm')
        test_images = [np.zeros((512, 512)),
                       np.ones((512, 512)),
                       np.random.rand(512, 512),
                       real_test_image]
        for i in range(len(test_images)):
            test_images[i] = np.repeat(test_images[i][..., np.newaxis], 3, -1)
        return np.array(test_images)

    def setUp(self):
        multi_model_path = os.path.join('models', 'categorical_model_six_full_improved_v5.h5')
        recurrent_model_path = os.path.join('models', 'recurrent_model_improved_v5.h5')
        self.multi_class_model = keras.models.load_model(multi_model_path, compile=True)
        self.recurrent_model = keras.models.load_model(recurrent_model_path, compile=True)
        self.test_images = self._init_test_images()
        self.test_labels = np.array([[0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 1, 0, 0]])
        self.multi_preds = self.multi_class_model.predict(self.test_images)
        reshaped_preds = self.multi_preds.reshape(1, *self.multi_preds.shape)
        self.recurrent_preds = self.recurrent_model.predict(reshaped_preds)

    def test_multi_predictions(self):
        is_within_range = np.all((self.multi_preds >= 0.0) & (self.multi_preds <= 1.0))
        self.assertTrue(is_within_range)

    def test_recurrent_predictions(self):
        is_within_range = np.all((self.recurrent_preds >= 0.0) & (self.recurrent_preds <= 1.0))
        self.assertTrue(is_within_range)

    def test_build_multi_class(self):
        expected_multi_model = self.multi_class_model.get_config()
        result_multi_model = StandardModel('xception', (512, 512, 3), classes=5, use_softmax=True)
        result_multi_model = result_multi_model.build_model().get_config()
        self.assertTrue(expected_multi_model, result_multi_model)

    def test_build_recurrent(self):
        expected_recurrent_model = self.recurrent_model.get_config()
        result_recurrent_model = StandardModel(classes=6)
        result_recurrent_model = result_recurrent_model.build_simple_recurrent_model().get_config()
        self.assertTrue(expected_recurrent_model, result_recurrent_model)

    def test_multi_weights_change(self):
        before_weights = self.multi_class_model.get_weights()
        self.multi_class_model.fit(self.test_images, self.test_labels,
                                   batch_size=1,
                                   epochs=1,
                                   verbose=0)
        after_weights = self.multi_class_model.get_weights()
        weights_constant = True
        for before, after in zip(before_weights, after_weights):
            weights_constant &= np.allclose(before, after)
        del before_weights
        del after_weights
        self.assertFalse(weights_constant)

    def test_recurrent_weights_change(self):
        before_weights = self.recurrent_model.get_weights()
        reshaped_preds = self.multi_preds.reshape(1, *self.multi_preds.shape)
        reshaped_labels = self.test_labels.reshape(1, *self.test_labels.shape)
        self.recurrent_model.fit(reshaped_preds, reshaped_labels,
                                 batch_size=1,
                                 epochs=1,
                                 verbose=0)
        after_weights = self.recurrent_model.get_weights()
        weights_constant = True
        for before, after in zip(before_weights, after_weights):
            weights_constant &= np.allclose(before, after)
        del before_weights
        del after_weights
        self.assertFalse(weights_constant)


if __name__ == '__main__':
    unittest.main()
