import tensorflow as tf


def create_rescaled_sigmoid_fn(min: float, max: float):
    def rescaled_sigmoid(x):
        return (max - min) * tf.sigmoid(x) + min

    return rescaled_sigmoid
