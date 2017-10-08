"""
Tensorflow implementation of attention model.
"""

import tensorflow as tf
import numpy as np

from attention_configuration import AttentionConfiguration, LearningConfiguration

# ----------------------------------------------------------------------------------------------------------------------
# Attention Model
# ----------------------------------------------------------------------------------------------------------------------


class AttentionModel(object):

    def __init__(self, attention_config, learning_config):
        """Initialize attention model.

        :param attention_config:
            Configuration object for specifying attention model.
        :param learning_config:
            Configuration object for specifying weight and bias initialization.
        """
        self._validate_attention_configuration(attention_config)
        self._validate_learning_configuration(learning_config)

        self._model_config = attention_config
        self._learning_config = learning_config

        # initialize LSTM for attention model
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self._model_config.hidden_state_dimension)

    @staticmethod
    def _validate_attention_configuration(configuration):
        assert isinstance(configuration, AttentionConfiguration)

    @staticmethod
    def _validate_learning_configuration(configuration):
        assert isinstance(configuration, LearningConfiguration)

    def get_model_inputs(self):
        """Initialize placeholder objects for tensorflow.

        On each batch iteration, this dictionary must be populated with the data to be processed.

        :return:
            Dictionary of placeholder objects for holding raw data on each tensorflow batch iteration.
        """
        # specify dimensions for batch data
        features_shape = (self._model_config.batch_size,
                          self._model_config.number_of_annotation_vectors,
                          self._model_config.annotation_vector_dimension)

        sequences_shape = (self._model_config.batch_size,
                           self._model_config.sequence_length,
                           self._model_config.vocabulary_size)

        labels_shape = self._model_config.batch_size

        # create place holders
        features = tf.placeholder(dtype=tf.float32, shape=features_shape)
        sequences = tf.placeholder(dtype=tf.float32, shape=sequences_shape)
        labels = tf.placeholder(dtype=tf.int32, shape=labels_shape)

        return {'features': features, 'sequences': sequences, 'labels': labels}

    def get_loss_op(self, model_inputs):
        """Return loss for model.

        :param model_inputs:
            Dictionary containing model inputs populated with batch data.
        :return:
            Loss for batch iteration.
        """
        logits = self._compute_logits(features=model_inputs['features'],
                                      sequences=model_inputs['sequences'])
        total_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model_inputs['labels'],
                                                                                  logits=logits))
        return total_loss / tf.to_float(self._model_config.batch_size)

    def _compute_logits(self, features, sequences):
        """Compute logits for a single batch iteration.

        :param features:
            Numpy array of size (L x D) representing convolutional annotation vectors.
            See model configuration for details for definition of L and D.
        :param sequences
            Sequences corresponding to batch data.

        :return:
            Logits for each training example of batch.
        """

        # Get initial LSTM states for each sequence in batch (N x H)
        memory_state, hidden_state = get_initial_lstm(features=features,
                                                      model_config=self._model_config,
                                                      learning_config=self._learning_config)

        # Get hidden states for each sequence in batch (N x H)
        for t in range(self._model_config.sequence_length):
            with tf.variable_scope('update_lstm', reuse=(t != 0)):
                _, (memory_state, hidden_state) = self.lstm_cell(inputs=sequences[:, t, :],
                                                                 state=[memory_state, hidden_state])

        # Compute attention probabilities for each sequence in batch conditioned on hidden states (N x L)
        attention_probabilities = compute_attention_probabilities(features=features,
                                                                  hidden_state=hidden_state,
                                                                  model_config=self._model_config,
                                                                  learning_config=self._learning_config,
                                                                  reuse=None)

        # Select context based on attention probabilities.
        context = select_context(features, attention_probabilities, model_config=self._model_config)  # (N x D)

        # get logits
        logits = decode_lstm(hidden_state=hidden_state,
                             context=context,
                             model_config=self._model_config,
                             learning_config=self._learning_config,
                             reuse=None)

        return logits


# ----------------------------------------------------------------------------------------------------------------------
# Attention Layers
# ----------------------------------------------------------------------------------------------------------------------


def get_initial_lstm(features, model_config, learning_config):
    """Returns initial state of LSTM conditioned on CNN annotation features.

    Note that we want to separately initialize the hidden state for each sequence since sequences are independent.
    We do not want the state information from a previous sequence to leak into the current sequence.

    :param features:
        Features extracted from CNN of dimension (N x L x D), where
            N = batch size
            L = number of annotation vectors per training sequence
            D = dimension of each annotation vector
    :param model_config:
        Configuration object for specifying attention model.
    :param learning_config:
        Configuration object for specifying weight and bias initialization.
    :return:
        initial hidden and memory state.
    """
    # specify dimension of weights and bias of layer
    weight_shape = (model_config.annotation_vector_dimension, model_config.hidden_state_dimension)  # (D x H)
    bias_shape = model_config.hidden_state_dimension

    # compute mean of features (N x L x D) -> (N x D)
    features_mean = tf.reduce_mean(features, axis=1)

    with tf.variable_scope('initial_lstm', reuse=False):

        # specify hidden state weight and bias for LSTM initialization
        w_h = tf.get_variable('w_h', shape=weight_shape, initializer=learning_config.weight_initializer)
        b_h = tf.get_variable('b_h', shape=bias_shape, initializer=learning_config.constant_initializer)

        # specify memory state weight and bias for LSTM initialization
        w_c = tf.get_variable('w_c', shape=weight_shape, initializer=learning_config.weight_initializer)
        b_c = tf.get_variable('b_c', shape=bias_shape, initializer=learning_config.constant_initializer)

        # get initial logits
        h_init_logits = tf.matmul(features_mean, w_h) + b_h
        c_init_logits = tf.matmul(features_mean, w_c) + b_c

        # get initial states
        h_init = tf.nn.tanh(h_init_logits)
        c_init = tf.nn.tanh(c_init_logits)

        return h_init, c_init


def attention_project_features(features, model_config, learning_config, reuse=False):
    """Apply weighted transformation to features.

     :param features:
        Features extracted from CNN of dimension (N x L x D), where
            N = batch size
            L = number of annotation vectors per training sequence
            D = dimension of each annotation vector
    :param model_config:
        Configuration object for specifying attention model.
    :param learning_config:
        Configuration object for specifying weight and bias initialization.
    :return
        Weighted transformation of features (N x L x D).
    """
    features_shape = (model_config.batch_size,
                      model_config.number_of_annotation_vectors,
                      model_config.annotation_vector_dimension)  # (N x L x D)
    flatten_shape = (-1, model_config.annotation_vector_dimension)  # converts tensor to to (NL x D)
    weight_shape = (model_config.annotation_vector_dimension, model_config.annotation_vector_dimension)  # (D x D)

    with tf.variable_scope('attention_project_features', reuse=reuse):

        # specify weight matrix
        w_features = tf.get_variable(
            name='w_features', shape=weight_shape, initializer=learning_config.weight_initializer) # (D x D)

        # flatten features and project
        features_flat = tf.reshape(tensor=features, shape=flatten_shape)  # (N x L x D) -> (NL x D)
        projected_features = tf.matmul(features_flat, w_features)  # (NL x D)

        # reshape to original dimensions
        projected_features = tf.reshape(tensor=projected_features, shape=features_shape)
        return projected_features


def attention_project_hidden_state(hidden_state, model_config, learning_config, reuse=False):
    """Apply weighted transformation to hidden state.

    :param hidden_state:
        Hidden state of LSTM.
    :param model_config:
        Configuration object for specifying attention model.
    :param learning_config:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        Weighted transformation on hidden state.
    """
    weight_shape = (model_config.hidden_state_dimension, model_config.annotation_vector_dimension)

    with tf.variable_scope('attention_project_hidden_state', reuse=reuse):
        w_hidden = tf.get_variable(name='w_hidden', shape=weight_shape, initializer=learning_config.weight_initializer)
        projected_h = tf.matmul(hidden_state, w_hidden)
        return projected_h


def attention_bias(model_config, learning_config, reuse=False):
    """Get attention bias.

    :param model_config:
        Configuration object for specifying attention model.
    :param learning_config:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        Bias vector.
    """
    with tf.variable_scope('attention_bias', reuse=reuse):
        bias = tf.get_variable(name='b',
                               shape=model_config.annotation_vector_dimension,
                               initializer=learning_config.constant_initializer)
        return bias


def process_attention_inputs(features, hidden_state, model_config, learning_config):
    """Apply transformations to raw inputs to attention model.

    This function performs weighted transformations on two raw input sources for the attention model.
        1. convolutional annotation vectors
        2. batch sequences
    The attention LSTM computes logits based on these transformed inputs.

    :param features:
        Features extracted from CNN of dimension (N x L x D), where
            N = batch size
            L = number of annotation vectors per training sequence
            D = dimension of each annotation vector
    :param hidden_state:
        Hidden state conditioned on CNN annotation vectors.
    :param model_config:
        Configuration object for specifying attention model.
    :param learning_config:
        Configuration object for specifying weight and bias initialization.
    :return:
        3-tuple of transformed inputs (features, hidden state, bias)
    """
    # transform features (N x L x D)
    projected_features = attention_project_features(features=features,
                                                    model_config=model_config,
                                                    learning_config=learning_config)

    # transform hidden state (N x 1 x D)
    projected_hidden_shape = (model_config.batch_size, 1, model_config.annotation_vector_dimension)
    projected_h = attention_project_hidden_state(hidden_state=hidden_state,
                                                 model_config=model_config,
                                                 learning_config=learning_config)
    projected_h = tf.transpose(projected_h)
    projected_h = tf.reshape(projected_h, shape=projected_hidden_shape)

    # get bias
    bias = attention_bias(model_config=model_config, learning_config=learning_config)

    return projected_features, projected_h, bias


def compute_attention_probabilities(features, hidden_state, model_config, learning_config, reuse=False):
    """Return attention probabilities.

    :param features:
        Features extracted from CNN of dimension (N x L x D), where
            N = batch size
            L = number of annotation vectors per training sequence
            D = dimension of each annotation vector
    :param hidden_state:
        Hidden state conditioned on CNN annotation vectors.
    :param model_config:
        Configuration object for specifying attention model.
    :param learning_config:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        (N x L) matrix of probabilities. For each sequence in batch, there are L corresponding annotation vectors.
        A probability is computed for each of these annotation vectors.
    """

    # specify weight dimensions
    attention_weight_shape = (model_config.annotation_vector_dimension, 1)
    attention_reshape_dimension = (-1, model_config.annotation_vector_dimension)  # converts (N x L x D) -> (NL x D)
    attention_logits_shape = (model_config.batch_size, model_config.number_of_annotation_vectors)  # (N x L)

    with tf.variable_scope('attention_layer', reuse=reuse):
        projected_features, projected_h, bias = process_attention_inputs(features=features,
                                                                         hidden_state=hidden_state,
                                                                         model_config=model_config,
                                                                         learning_config=learning_config)

        # create attention input
        # note that +bias is a broadcasted operation
        attention_input = projected_features + projected_h + bias  # (N x L x D)
        attention_input = tf.reshape(attention_input, shape=attention_reshape_dimension)  # (NL x D)

        # compute attention logits
        w_attention = tf.get_variable(name='w_attention',
                                      shape=attention_weight_shape,
                                      initializer=learning_config.weight_initializer)  # (D x 1)
        attention_logits = tf.matmul(attention_input, w_attention)  # (NL x 1)
        attention_logits = tf.reshape(attention_logits, shape=attention_logits_shape)  # (N x L)

        # compute attention probabilities
        attention_probabilities = tf.nn.softmax(attention_logits)  # (N x L)
        return attention_probabilities


def select_context(features, attention_probabilities, model_config):
    """Select context vector from attention probabilities

    :param features:
        Features extracted from CNN of dimension (N x L x D), where
            N = batch size
            L = number of annotation vectors per training sequence
            D = dimension of each annotation vector
    :param attention_probabilities:
        (N x L) tensor of probabilities.
    :return:
        (N x D), where each row represents the context vector selected for ith example.
    """
    selected_context_indices = tf.argmax(attention_probabilities, axis=1)
    gather_indices = convert_to_gather_indices(selected_context_indices, model_config)
    return tf.gather_nd(params=features, indices=gather_indices)


def decode_lstm(hidden_state, context, model_config, learning_config, reuse=False):
    """Get predictions given hidden state and context for batch.

    :param hidden_state:
        Hidden state for LSTM computed from batch sequences and CNN annotation vectors.
    :param context:
        Selected context vectors.
    :param model_config:
        Configuration object for specifying attention model.
    :param learning_config:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        Logits for attention model.
    """
    # specify dimensions of weights and biases
    hidden_weight_shape = (model_config.hidden_state_dimension, model_config.prediction_classes)
    context_weight_shape = (model_config.annotation_vector_dimension, model_config.prediction_classes)

    with tf.variable_scope('decode_lstm', reuse=reuse):
        w_hidden = tf.get_variable(name='w_hidden',
                                   shape=hidden_weight_shape,
                                   initializer=learning_config.weight_initializer)

        w_context = tf.get_variable(name='w_context',
                                    shape=context_weight_shape,
                                    initializer=learning_config.weight_initializer)

        bias_out = tf.get_variable('bias_out',
                                   shape=model_config.prediction_classes,
                                   initializer=learning_config.constant_initializer)

        hidden_contribution = tf.matmul(hidden_state, w_hidden)
        context_contribution = tf.matmul(context, w_context)
        logits = hidden_contribution + context_contribution + bias_out
        return logits


# ----------------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------------


def convert_to_gather_indices(selected_indices, model_config):
    """Convert selected context indices to tensor to be used for gather_nd."""
    indices = tf.reshape(np.arange(model_config.batch_size), shape=(model_config.batch_size, 1))
    selected_context_indices = tf.reshape(selected_indices, shape=(model_config.batch_size, 1))
    return tf.concat((indices, selected_context_indices), axis=1)
