import tensorflow as tf

layers = tf.keras.layers


class DenseBlock(layers.Layer):
    def __init__(self, num_outputs: int, dropout: float):
        """
        A dense block is a composition of some layers
        :param num_outputs: The number of neurons in the dense layer
        :param dropout: The fraction of inputs to drop
        """
        super(DenseBlock, self).__init__()
        self.dense = layers.Dense(num_outputs, activation='relu')
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, input):
        x = self.dense(input)
        x = self.bn(x)
        x = self.dropout(x)

        return x
