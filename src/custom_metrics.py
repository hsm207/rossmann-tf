import tensorflow as tf


def root_mean_squared_percentage_error(y_true, y_pred):
    # assume output from model is log scale and label is unscaled
    y_pred = tf.exp(y_pred)
    squared_prct = tf.square((y_pred - y_true)/y_true)

    return tf.sqrt(tf.reduce_mean(squared_prct))


rmspe = root_mean_squared_percentage_error
