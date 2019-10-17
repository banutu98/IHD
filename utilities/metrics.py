import numpy as np
from keras import backend as K


def weighted_log_loss(y_true, y_pred, weights):
    class_weights = np.array(weights)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.0 - eps)
    out = -(y_true * K.log(y_pred) * class_weights +
            (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)
    return K.mean(out, axis=-1)
