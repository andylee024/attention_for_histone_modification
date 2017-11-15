import tensorflow as tf
import numpy as np

from komorebi.libs.model.abstract_model import AbstractTensorflowModel
from komorebi.libs.model.attention_configuration import AttentionConfiguration
from komorebi.libs.model.parameter_initialization import ParameterInitializationPolicy


class AttentionModel(AbstractTensorflowModel):

    def __init__(self, attention_config, parameter_policy):
        """Initialize attention model.

        :param attention_config:
            Configuration object for specifying attention model.
        :param parameter_policy:
            Policy specifying weight and bias initialization.
        """
        assert isinstance(attention_config, AttentionConfiguration)
        assert isinstance(parameter_policy, ParameterInitializationPolicy)
        self._model_config = attention_config
        self._parameter_policy = parameter_policy

        # initialize LSTM for attention model
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self._model_config.hidden_state_dimension)

    def predict(self, features, sequences):
       """Compute predictions for attention model.

       :param features: tf.placeholder populated with convolutional annotation vectors.
       :param sequences: tf.placeholder populated with sequence data.
       :return: logits corresponding to prediction.
       """
       # Get initial LSTM states for each sequence in batch (N x H)
       memory_state, hidden_state = get_initial_lstm(features=features,
                                                     model_config=self._model_config,
                                                     parameter_policy=self._parameter_policy)

       # Get hidden states for each sequence in batch (N x H)
       for t in range(self._model_config.sequence_length):
           with tf.variable_scope('update_lstm', reuse=(t != 0)):
               _, (memory_state, hidden_state) = self.lstm_cell(inputs=sequences[:, t, :],
                                                                state=[memory_state, hidden_state])

       # Compute attention probabilities for each sequence in batch conditioned on hidden states (N x L)
       attention_probabilities = compute_attention_probabilities(features=features,
                                                                 hidden_state=hidden_state,
                                                                 model_config=self._model_config,
                                                                 parameter_policy=self._parameter_policy,
                                                                 reuse=None)

       # Select context based on attention probabilities.
       context = select_context(features, attention_probabilities, model_config=self._model_config)  # (N x D)

       # get logits
       logits = decode_lstm(hidden_state=hidden_state,
                            context=context,
                            model_config=self._model_config,
                            parameter_policy=self._parameter_policy,
                            reuse=None)
       return logits
   
    @property
    def inputs(self):
        """Return tf.placeholders for holding inputs of model.

        :return:
            Dictionary with following attributes.
                "features"  :   placeholder of shape (batch_size, number_annotations, annotation_size)
                "sequences" :   placeholder of shape (batch_size, sequence_length, vocabulary_size)
        """
        features_shape = (None, self._model_config.number_of_annotations, self._model_config.annotation_size)
        sequences_shape = (None, self._model_config.sequence_length, self._model_config.vocabulary_size)

        return {'features'  : tf.placeholder(dtype=tf.float32, shape=features_shape),
                'sequences' : tf.placeholder(dtype=tf.float32, shape=sequences_shape)}

    @property
    def outputs(self):
        """Return tf.placeholders for holding outputs of model.

        :return:
            Dictionary with following attributes.
                "labels" : placeholder of shape (batch_size, prediction_classes)
        """
        labels_shape = (None, self._model_config.prediction_classes)
        return {'labels' : tf.placeholder(dtype=tf.float32, shape=labels_shape)}



# ----------------------------------------------------------------------------------------------------------------------
# Attention Layers
# ----------------------------------------------------------------------------------------------------------------------

def get_initial_lstm(features, model_config, parameter_policy):
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
    :param parameter_policy:
        Configuration object for specifying weight and bias initialization.
    :return:
        initial hidden and memory state.
    """
    # specify dimension of weights and bias of layer
    weight_shape = (model_config.annotation_size, model_config.hidden_state_dimension)  # (D x H)
    bias_shape = model_config.hidden_state_dimension

    # compute mean of features (N x L x D) -> (N x D)
    features_mean = tf.reduce_mean(features, axis=1)

    with tf.variable_scope('initial_lstm', reuse=False):

        # specify hidden state weight and bias for LSTM initialization
        w_h = tf.get_variable('w_h', shape=weight_shape, initializer=parameter_policy.weight_initializer)
        b_h = tf.get_variable('b_h', shape=bias_shape, initializer=parameter_policy.constant_initializer)

        # specify memory state weight and bias for LSTM initialization
        w_c = tf.get_variable('w_c', shape=weight_shape, initializer=parameter_policy.weight_initializer)
        b_c = tf.get_variable('b_c', shape=bias_shape, initializer=parameter_policy.constant_initializer)

        # get initial logits
        h_init_logits = tf.matmul(features_mean, w_h) + b_h
        c_init_logits = tf.matmul(features_mean, w_c) + b_c

        # get initial states
        h_init = tf.nn.tanh(h_init_logits)
        c_init = tf.nn.tanh(c_init_logits)

        return h_init, c_init


def attention_project_features(features, model_config, parameter_policy, reuse=False):
    """Apply weighted transformation to features.

     :param features:
        Features extracted from CNN of dimension (N x L x D), where
            N = batch size
            L = number of annotation vectors per training sequence
            D = dimension of each annotation vector
    :param model_config:
        Configuration object for specifying attention model.
    :param parameter_policy:
        Configuration object for specifying weight and bias initialization.
    :return
        Weighted transformation of features (N x L x D).
    """
    features_shape = (get_batch_size(features),
                      model_config.number_of_annotations,
                      model_config.annotation_size)  # (N x L x D)
    flatten_shape = (-1, model_config.annotation_size)  # converts tensor to to (NL x D)
    weight_shape = (model_config.annotation_size, model_config.annotation_size)  # (D x D)

    with tf.variable_scope('attention_project_features', reuse=reuse):

        # specify weight matrix
        w_features = tf.get_variable(
            name='w_features', shape=weight_shape, initializer=parameter_policy.weight_initializer) # (D x D)

        # flatten features and project
        features_flat = tf.reshape(tensor=features, shape=flatten_shape)  # (N x L x D) -> (NL x D)
        projected_features = tf.matmul(features_flat, w_features)  # (NL x D)

        # reshape to original dimensions
        projected_features = tf.reshape(tensor=projected_features, shape=features_shape)
        return projected_features


def attention_project_hidden_state(hidden_state, model_config, parameter_policy, reuse=False):
    """Apply weighted transformation to hidden state.

    :param hidden_state:
        Hidden state of LSTM.
    :param model_config:
        Configuration object for specifying attention model.
    :param parameter_policy:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        Weighted transformation on hidden state.
    """
    weight_shape = (model_config.hidden_state_dimension, model_config.annotation_size)

    with tf.variable_scope('attention_project_hidden_state', reuse=reuse):
        w_hidden = tf.get_variable(name='w_hidden', shape=weight_shape, initializer=parameter_policy.weight_initializer)
        projected_h = tf.matmul(hidden_state, w_hidden)
        return projected_h


def attention_bias(model_config, parameter_policy, reuse=False):
    """Get attention bias.

    :param model_config:
        Configuration object for specifying attention model.
    :param parameter_policy:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        Bias vector.
    """
    with tf.variable_scope('attention_bias', reuse=reuse):
        bias = tf.get_variable(name='b',
                               shape=model_config.annotation_size,
                               initializer=parameter_policy.constant_initializer)
        return bias


def process_attention_inputs(features, hidden_state, model_config, parameter_policy):
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
    :param parameter_policy:
        Configuration object for specifying weight and bias initialization.
    :return:
        3-tuple of transformed inputs (features, hidden state, bias)
    """
    # transform features (N x L x D)
    projected_features = attention_project_features(features=features,
                                                    model_config=model_config,
                                                    parameter_policy=parameter_policy)

    # transform hidden state (N x 1 x D)
    projected_hidden_shape = (get_batch_size(features), 1, model_config.annotation_size)
    projected_h = attention_project_hidden_state(hidden_state=hidden_state,
                                                 model_config=model_config,
                                                 parameter_policy=parameter_policy)
    projected_h = tf.transpose(projected_h)
    projected_h = tf.reshape(projected_h, shape=projected_hidden_shape)

    # get bias
    bias = attention_bias(model_config=model_config, parameter_policy=parameter_policy)

    return projected_features, projected_h, bias


def compute_attention_probabilities(features, hidden_state, model_config, parameter_policy, reuse=False):
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
    :param parameter_policy:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        (N x L) matrix of probabilities. For each sequence in batch, there are L corresponding annotation vectors.
        A probability is computed for each of these annotation vectors.
    """

    # specify weight dimensions
    attention_weight_shape = (model_config.annotation_size, 1)
    attention_reshape_dimension = (-1, model_config.annotation_size)  # converts (N x L x D) -> (NL x D)
    attention_logits_shape = (get_batch_size(features), model_config.number_of_annotations)  # (N x L)

    with tf.variable_scope('attention_layer', reuse=reuse):
        projected_features, projected_h, bias = process_attention_inputs(features=features,
                                                                         hidden_state=hidden_state,
                                                                         model_config=model_config,
                                                                         parameter_policy=parameter_policy)

        # create attention input
        # note that +bias is a broadcasted operation
        attention_input = projected_features + projected_h + bias  # (N x L x D)
        attention_input = tf.reshape(attention_input, shape=attention_reshape_dimension)  # (NL x D)

        # compute attention logits
        w_attention = tf.get_variable(name='w_attention',
                                      shape=attention_weight_shape,
                                      initializer=parameter_policy.weight_initializer)  # (D x 1)
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
    selected_context_indices = tf.argmax(attention_probabilities, axis=1, output_type=tf.int32)
    gather_indices = convert_to_gather_indices(selected_context_indices, batch_size=get_batch_size(features))
    return tf.gather_nd(params=features, indices=gather_indices)


def decode_lstm(hidden_state, context, model_config, parameter_policy, reuse=False):
    """Get predictions given hidden state and context for batch.

    :param hidden_state:
        Hidden state for LSTM computed from batch sequences and CNN annotation vectors.
    :param context:
        Selected context vectors.
    :param model_config:
        Configuration object for specifying attention model.
    :param parameter_policy:
        Configuration object for specifying weight and bias initialization.
    :param reuse:
        Bool. Use existing weight matrix if true. Use new weight matrix if false.
    :return:
        Logits for attention model.
    """
    # specify dimensions of weights and biases
    hidden_weight_shape = (model_config.hidden_state_dimension, model_config.prediction_classes)
    context_weight_shape = (model_config.annotation_size, model_config.prediction_classes)

    with tf.variable_scope('decode_lstm', reuse=reuse):
        w_hidden = tf.get_variable(name='w_hidden',
                                   shape=hidden_weight_shape,
                                   initializer=parameter_policy.weight_initializer)

        w_context = tf.get_variable(name='w_context',
                                    shape=context_weight_shape,
                                    initializer=parameter_policy.weight_initializer)

        bias_out = tf.get_variable('bias_out',
                                   shape=model_config.prediction_classes,
                                   initializer=parameter_policy.constant_initializer)

        hidden_contribution = tf.matmul(hidden_state, w_hidden)
        context_contribution = tf.matmul(context, w_context)
        logits = hidden_contribution + context_contribution + bias_out
        return logits


# ----------------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------------


def convert_to_gather_indices(selected_indices, batch_size):
    """Convert selected context indices to tensor to be used for gather_nd."""
    indices = tf.reshape(tf.range(batch_size), shape=(-1, 1))
    selected_context_indices = tf.reshape(selected_indices, shape=(-1, 1))
    return tf.concat([indices, selected_context_indices], axis=1)

def get_batch_size(tensor):
    """Return batch size of tensor."""
    return tf.shape(tensor)[0]
