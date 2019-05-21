import tensorflow as tf


def mean_squared_percentage_error(y_true, y_pred):
    return tf.reduce_sum(tf.square(tf.divide(y_true - y_pred, y_true)))
