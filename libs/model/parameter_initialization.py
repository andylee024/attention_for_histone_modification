import tensorflow as tf

class ParameterInitializationPolicy(object):
    """Policy specifying how to initialize parameters."""

    def __init__(self):
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.constant_initializer = tf.constant_initializer(0.0)

