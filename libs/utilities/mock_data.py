import collections
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Mock data structures for ML
# ----------------------------------------------------------------------------------------------------------------------

TrainingExample = collections.namedtuple(
    typename='TrainingExample', field_names=['sequence', 'annotation', 'label'])

TrainingTensor = collections.namedtuple(
    typename="TrainingTensor", field_names=['sequence_tensor', 'annotation_tensor', 'label_tensor'])

# ----------------------------------------------------------------------------------------------------------------------
# Public APIs for creating mock data
# ----------------------------------------------------------------------------------------------------------------------
def create_dummy_sequence(sequence_length=100, vocabulary_size=4, one_hot=True):
    """Create dummy sequence.
    
    :param sequence_length: length of sequence.
    :param vocabulary_size: size of vocabulary.
    :param one_hot: If true, then return one-hot representation.
    :return: Numpy array representing sequence.
    """
    dummy_sequence = np.random.randint(low=0, high=vocabulary_size, size=sequence_length)
    return convert_to_one_hot(labels=dummy_sequence, number_of_classes=vocabulary_size) if one_hot else dummy_sequence

def create_dummy_label(prediction_classes, one_hot=True):
    """Create dummy label.

    :param prediction_classes: number of prediction classes.
    :return: Numpy array representing label. 
    """
    label = np.random.randint(low=0, high=prediction_classes, size=1)
    return convert_label_to_one_hot(label, prediction_classes) if one_hot else label

def create_dummy_annotation(number_of_annotations, annotation_dimension):
    """Create dummy annotation.

    :param number_of_annotations: Int. Number of annotations associated with a single sample.
    :param annotation_dimension: Int. Dimension of each annotation.
    :return: Numpy array representing annotation.
    """
    annotation_vector_size = (number_of_annotations, annotation_dimension)
    return np.random.normal(loc=0.0, scale=1.0, size=annotation_vector_size)

def create_dummy_sequence_batch(sequence_length=100, vocabulary_size=4, batch_size=100):
    """Create batch of sequences."""
    sequences = [np.expand_dims(create_dummy_sequence(sequence_length, vocabulary_size), axis=0) for _ in xrange(batch_size)]
    return np.concatenate(sequences, axis=0)

def create_dummy_label_batch(prediction_classes, batch_size=100):
    """Create batch of one-hot encoded labels."""
    labels = [create_dummy_label(prediction_classes) for _ in xrange(batch_size)]
    return np.concatenate(labels, axis=0)

def create_dummy_annotation_batch(number_of_annotations, annotation_dimension, batch_size=100):
    """Create batch of annotations."""
    annotations = [np.expand_dims(create_dummy_annotation(number_of_annotations, annotation_dimension), axis=0) for _ in xrange(batch_size)]
    return np.concatenate(annotations, axis=0)

def create_dummy_training_example(attention_config):
    """Create a single training example with dummy data according to attention configuration object."""

    dummy_sequence = create_dummy_sequence(sequence_length=attention_config.sequence_length,
                                           vocabulary_size=attention_config.vocabulary_size,
                                           one_hot=True)

    dummy_label = create_dummy_label(prediction_classes=attention_config.prediction_classes,
                                     one_hot=True)

    dummy_annotation = create_dummy_annotation(number_of_annotations=attention_config.number_of_annotations,
                                               annotation_dimension=attention_config.annotation_dimension)
                                                
    return TrainingExample(sequence=dummy_sequence,
                           annotation_vectors=dummy_annotation,
                           label=dummy_label)

def create_dummy_batch_data(attention_config):
    """Create training examples for batch.

    ;return:
        Tensor representation of training examples.
    """
    training_examples = [create_dummy_training_example(attention_config) for _ in xrange(attention_config.batch_size)]
    return convert_training_examples_to_tensor(training_examples)


# ----------------------------------------------------------------------------------------------------------------------
# One-hot utilities
# ----------------------------------------------------------------------------------------------------------------------

def convert_label_to_one_hot(label, number_of_classes):
    """Converts a discrete label to one hot encoding.

    @param label: integer value representing label to encode
    @param number_of_classes: number of label classes
    @return: 1xC one-hot encoding, where C is number of classes
    """
    one_hot_encoding = np.zeros(number_of_classes)
    one_hot_encoding[label] = 1
    return np.reshape(one_hot_encoding, newshape=(1, number_of_classes))


def convert_to_one_hot(labels, number_of_classes):
    """Convert a list of labels to one hot encoding.

    @param labels: numpy array of discrete labels
    @param number_of_classes: number of label classes
    @return: one-hot encoding of labels (N x C), where N is batch size, C is number of classes
    """
    one_hot_labels = [convert_label_to_one_hot(l, number_of_classes=number_of_classes) for l in labels]
    return np.concatenate(one_hot_labels, axis=0)


def convert_training_examples_to_tensor(training_examples):
    """Convert batch data to tensor representation.

    @param training_examples:
        List of training examples.
    @return:
        Numpy tensors of dimension (N x (A1 x A2)), where N is batch dimension and (A1 x A2) is dimension of
        matrix corresponding to training example
    """
    # Add batch dimension to tensors and concatenate matrices across batch dimension
    # np.expand_dims - adds batch dimension
    # np.concatenate - creates batch tensor by stacking on batch dimension
    sequence_tensor = np.concatenate([np.expand_dims(te.sequence, axis=0) for te in training_examples], axis=0)
    annotation_tensor = np.concatenate([np.expand_dims(te.annotation_vectors, axis=0) for te in training_examples],
                                       axis=0)
    label_tensor = np.concatenate([np.expand_dims(te.label, axis=0) for te in training_examples], axis=0)
    label_tensor = np.ravel(label_tensor)

    return TrainingTensor(sequence_tensor=sequence_tensor,
                          annotation_tensor=annotation_tensor,
                          label_tensor=label_tensor)
