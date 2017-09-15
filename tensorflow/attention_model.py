"""
Tensorflow implementation of attention model for histone modification.
"""

import tensorflow as tf
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Attention Configuration
# ----------------------------------------------------------------------------------------------------------------------


class AttentionConfiguration(object):
    """Configuration data structure containing input parameters to attention model."""

    def __init__(self,
                 batch_size=100,
                 sequence_length=400,
                 vocabulary_size=4,
                 prediction_classes=3,
                 annotation_matrix_size=(196, 500),
                 hidden_state_dimension=100):
        """Initialize data structure.

        :param annotation_matrix_size:
            2-tuple representing L x D, where L is number of annotation vectors
            per training example and D is dimension of each annotation vector.
        """

        self.N = batch_size
        self.T = sequence_length
        self.V = vocabulary_size
        self.C = prediction_classes
        self.L, self.D = annotation_matrix_size
        self.H = hidden_state_dimension


class LearningConfiguration(object):
    """Initialization data structure."""

    def __init__(self):
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.constant_initializer = tf.constant_initializer(0.0)


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
        self._validate_configuration(attention_config)
        self._model_config = attention_config
        self._learning_config = learning_config

    @staticmethod
    def _validate_configuration(configuration):
        assert isinstance(configuration, AttentionConfiguration)

    def get_inputs(self):
        # inputs
        features = tf.placeholder(tf.float32, (None, self._model_config.L, self._model_config.D))
        sequences = tf.placeholder(tf.float32, (None, self._model_config.T, self._model_config.V))
        labels = tf.placeholder(tf.int32, None)

        return {'features': features, 'sequences': sequences, 'labels': labels}

    def get_loss(self, model_inputs):
        loss = 0.0

        # inputs
        features = model_inputs['features']
        sequences = model_inputs['sequences']
        labels = model_inputs['labels']

        # initialization
        c, h = get_initial_lstm(features,
                                model_config=self._model_config,
                                learning_config=self._learning_config,
                                reuse=False)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self._model_config.H)

        # process sequence to get updated hidden unit
        for t in range(self._model_config.T):
            with tf.variable_scope('update_lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=sequences[:, t, :], state=[c, h])

        # get context
        attention_probabilities = attention_layer(features=features,
                                                  h=h,
                                                  model_config=self._model_config,
                                                  learning_config=self._learning_config,
                                                  reuse=None)  # (N x L)
        context = select_context(features, attention_probabilities, model_config=self._model_config)  # (N x D)

        # get logits
        logits = decode_lstm(hidden_state=h, context=context, model_config=self._model_config, learning_config=self._learning_config, reuse=None)

        # get loss
        loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # return loss
        return loss / tf.to_float(self._model_config.N)

# ----------------------------------------------------------------------------------------------------------------------
# Attention Layers
# ----------------------------------------------------------------------------------------------------------------------


def get_initial_lstm(features, model_config, learning_config, reuse=False):
    """Returns initial state of LSTM by initializing with CNN features.

    Input: features (N x L x D)
    Output: hidden_state (N x H), memory_state (N x H)

    Note that we want to separately initialize the hidden state for each sequence
    because we assume that the sequences are independent. We do
    not want the state information from a previous sequence to leak into the current sequence.

    :param features:
        Features extracted from CNN of dimension (L x D).
    :return:
        initial hidden and memory state.
    """
    features_mean = tf.reduce_mean(features, axis=1)  # (N x D)

    with tf.variable_scope('initial_lstm', reuse=reuse):

        # get initial hidden state
        w_h = tf.get_variable('w_h',
                              shape=(model_config.D, model_config.H),
                              initializer=learning_config.weight_initializer)
        b_h = tf.get_variable('b_h',
                              shape=model_config.H,
                              initializer=learning_config.constant_initializer)
        h_init_logits = tf.matmul(features_mean, w_h) + b_h
        h_init = tf.nn.tanh(h_init_logits)

        # get initial memory state
        w_c = tf.get_variable('w_c',
                              shape=(model_config.D, model_config.H),
                              initializer=learning_config.weight_initializer)
        b_c = tf.get_variable('b_c',
                              shape=model_config.H,
                              initializer=learning_config.constant_initializer)

        c_init_logits = tf.matmul(features_mean, w_c) + b_c
        c_init = tf.nn.tanh(c_init_logits)

        return h_init, c_init


def attention_project_features(features, model_config, learning_config, reuse=False):
    """Apply weighted transformation to features.

    Input: (N x L x D) - all annotation vectors for each batch entry
    Output: (N x L x D) - projected annotation vectors for each batch entry
    """
    with tf.variable_scope('attention_project_features', reuse=reuse):
        features_flat = tf.reshape(features, [-1, model_config.D])  # (NL x D)
        w_features = tf.get_variable('w_features', [model_config.D, model_config.D], initializer=learning_config.weight_initializer)
        projected_features = tf.matmul(features_flat, w_features)  # (NL x D)
        projected_features = tf.reshape(projected_features,
                                        [model_config.N, model_config.L, model_config.D])  # (N x L x D)
        return projected_features


def attention_project_hidden_state(h, model_config, learning_config, reuse=False):
    """Apply weighted transformation to hidden state.

    Input: (N x H) - hidden state for each batch entry
    Output: (N x D) - projected hidden state for each batch entry
    """
    with tf.variable_scope('attention_project_hidden_state', reuse=reuse):
        w_hidden = tf.get_variable('w_hidden', [model_config.H, model_config.D], initializer=learning_config.weight_initializer)
        projected_h = tf.matmul(h, w_hidden)
        return projected_h


def attention_bias(model_config, learning_config, reuse=False):
    """Get attention bias.

    Output: (H x 1)
    """
    with tf.variable_scope('attention_bias', reuse=reuse):
        b = tf.get_variable('b', [model_config.D], initializer=learning_config.constant_initializer)
        return b


def get_attention_inputs(features, h, model_config, learning_config):
    # transform features (N x L x D)
    projected_features = attention_project_features(features=features,
                                                    model_config=model_config,
                                                    learning_config=learning_config)

    # transform hidden state (N x 1 x D)
    projected_h = tf.transpose(
        attention_project_hidden_state(h=h, model_config=model_config, learning_config=learning_config))
    projected_h = tf.reshape(projected_h, shape=[model_config.N, 1, model_config.D])

    # get bias
    bias = attention_bias(model_config=model_config, learning_config=learning_config)

    return projected_features, projected_h, bias


def attention_layer(features, h, model_config, learning_config, reuse=False):
    """Returns attention probabilities.

    Input:
        1. (N x L x D) - features for each batch entry
        2. (N x H) - hidden state for each batch entrying respective sequences.

    Output:
        1. (N x L) matrix of probabilities for each annotation vector in each batch entry
    """
    with tf.variable_scope('attention_layer', reuse=reuse):
        projected_features, projected_h, bias = get_attention_inputs(features=features, h=h, model_config=model_config, learning_config=learning_config)

        # create attention input
        # note that +bias is a broadcasted operation
        attention_input = projected_features + projected_h  # + bias # (N x L x D)
        attention_input = tf.reshape(attention_input, shape=[-1, model_config.D])  # (NL x D)

        # apply attention mechanism
        w_attention = tf.get_variable('w_attention', shape=[model_config.D, 1], initializer=learning_config.weight_initializer)
        attention_logits = tf.matmul(attention_input, w_attention)  # (NL x 1)
        attention_logits = tf.reshape(attention_logits, shape=(model_config.N, model_config.L))  # (N x L)

        # compute attention probabilities
        attention_probabilities = tf.nn.softmax(attention_logits)  # (N x L)
        return attention_probabilities


def convert_to_gather_indices(selected_indices, model_config):
    """Convert selected context indices to tensor to be used for gather_nd."""
    indices = tf.reshape(np.arange(model_config.N), shape=(model_config.N, 1))
    selected_context_indices = tf.reshape(selected_indices, shape=(model_config.N, 1))
    return tf.concat((indices, selected_context_indices), axis=1)


def select_context(features, attention_probabilities, model_config):
    """Select context vector from attention probabilities

    :param features:
        (N x L x D) tensor, where N is batch size, L is number of attention
        vectors and D is dimension of attention vector.
    :param attention_probabilities:
        (N x L) tensor of probabilities.
    :return:
        (N x D), where each row represents a context vector for ith example.
    """
    selected_context_indices = tf.argmax(attention_probabilities, axis=1)
    gather_indices = convert_to_gather_indices(selected_context_indices, model_config)
    return tf.gather_nd(params=features, indices=gather_indices)


def decode_lstm(hidden_state, context, model_config, learning_config, reuse=False):
    """Predict on hidden state and context."""
    with tf.variable_scope('decode_lstm', reuse=reuse):
        w_hidden = tf.get_variable('w_hidden', [model_config.H, model_config.C], initializer=learning_config.weight_initializer)
        w_context = tf.get_variable('w_context', [model_config.D, model_config.C], initializer=learning_config.weight_initializer)
        b_out = tf.get_variable('b_out', [model_config.C], initializer=learning_config.constant_initializer)

        hidden_contribution = tf.matmul(hidden_state, w_hidden)
        context_contribution = tf.matmul(context, w_context)
        logits = hidden_contribution + context_contribution + b_out
        return logits
