import tensorflow as tf


def mean_squared_percentage_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.divide(y_true - y_pred, y_true)))

def mse_log(y_true, y_pred):
    # assume y_pred is already at the
    y_true = tf.math.log(y_true + 1/1000)

    return tf.reduce_mean(tf.square(y_pred - y_true))