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


def create_dummy_batch_data(attention_config):
    """Create training examples for batch.

    ;return:
        Tensor representation of training examples.
    """
    training_examples = [create_dummy_training_example(
        attention_config) for _ in xrange(attention_config.batch_size)]
    return convert_training_examples_to_tensor(training_examples)


def create_dummy_training_example(attention_config):
    """Create a single training example with dummy data according to attention configuration object."""
    return TrainingExample(sequence=create_dummy_training_sequence(attention_config, one_hot=True),
                           label=create_dummy_training_label(
                               attention_config, one_hot=True),
                           annotation=create_dummy_training_annotation(attention_config))


def create_dummy_training_sequence(attention_config, one_hot=True):
    """Create dummy training sequence from attention config."""
    dummy_sequence = np.random.randint(low=0,
                                       high=attention_config.vocabulary_size,
                                       size=attention_config.sequence_length)
    if one_hot:
        dummy_sequence = convert_to_one_hot(labels=dummy_sequence,
                                            number_of_classes=attention_config.vocabulary_size)
    return dummy_sequence


def create_dummy_training_label(attention_config, one_hot=True):
    """Create dummy training label from attention config."""
    dummy_label = np.random.randint(
        low=0, high=attention_config.prediction_classes, size=1)
    if one_hot:
        dummy_label = convert_to_one_hot(
            labels=dummy_label,
            number_of_classes=attention_config.prediction_classes)
    return dummy_label


def create_dummy_training_annotation(attention_config):
    """Create dummy training annotation from attention config."""
    return np.random.normal(loc=0.0,
                            scale=1.0,
                            size=(attention_config.number_of_annotations, attention_config.annotation_size))

# ----------------------------------------------------------------------------------------------------------------------
# Mock data
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
    one_hot_labels = [convert_label_to_one_hot(
        l, number_of_classes=number_of_classes) for l in labels]
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
    sequence_tensor = np.concatenate(
        [np.expand_dims(te.sequence, axis=0) for te in training_examples], axis=0)
    annotation_tensor = np.concatenate(
        [np.expand_dims(te.annotation, axis=0) for te in training_examples], axis=0)
    label_tensor = np.concatenate(
        [te.label for te in training_examples], axis=0)

    return TrainingTensor(sequence_tensor=sequence_tensor,
                          annotation_tensor=annotation_tensor,
                          label_tensor=label_tensor)
