import tensorflow as tf


def mean_squared_percentage_error(y_true, y_pred):
    return tf.reduce_sum(tf.square(tf.divide(y_true - y_pred, y_true)))

def mse_log_log(y_true, y_pred):
    y_true = tf.math.log(y_true + 1/1000)
    y_pred = tf.math.log(y_pred + 1/1000)

    return tf.reduce_sum(tf.square(y_pred - y_true))