"""Configuration objects for attention model."""

import tensorflow as tf


class LearningConfiguration(object):
    """Configuration object to initialize weights, biases for deep learning models."""

    def __init__(self):
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.constant_initializer = tf.constant_initializer(0.0)


class AttentionConfiguration(object):
    """Configuration object containing parameters to attention model."""

    def __init__(self,
                 batch_size,
                 sequence_length,
                 vocabulary_size,
                 prediction_classes,
                 number_of_annotation_vectors,
                 annotation_vector_dimension,
                 hidden_state_dimension):
        """Initialize configuration.

        :param batch_size:
            Int. Number of training examples in batch.
        :param sequence_length:
            Int. Number of characters in sequence corresponding to one training example.
        :param vocabulary_size:
            Int. Size of sequence vocabulary (e.g. 'a', 'c', 'g', 't')
        :param prediction_classes:
            Int. Number of classes for classification problem.
        :param number_of_annotation_vectors:
            Int. Number of annotation vectors for each training sequence.
        :param annotation_vector_dimension:
            Int. Dimension of each annotation vector.
        :param hidden_state_dimension:
            Number of hidden units in LSTM used for attention model.
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.prediction_classes = prediction_classes
        self.number_of_annotation_vectors = number_of_annotation_vectors
        self.annotation_vector_dimension = annotation_vector_dimension
        self.hidden_state_dimension = hidden_state_dimension

        # Convenience accessors for backwards compatibility
        # self.N = batch_size
        # self.T = sequence_length
        # self.V = vocabulary_size
        # self.C = prediction_classes
        # self.L = number_of_annotation_vectors
        # self.D = annotation_vector_dimension
        # self.H = hidden_state_dimension
