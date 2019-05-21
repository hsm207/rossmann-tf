import custom_losses
import tensorflow as tf


def root_mean_squared_percentage_error(y_true, y_pred):
    return tf.sqrt(custom_losses.mean_squared_percentage_error(y_true, y_pred))


rmspe = root_mean_squared_percentage_error
