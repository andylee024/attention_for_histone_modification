
class AttentionConfiguration(object):
    """Configuration object containing parameters to attention model."""

    def __init__(self,
                 batch_size,
                 sequence_length,
                 vocabulary_size,
                 prediction_classes,
                 number_of_annotations,
                 annotation_size,
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
        :param number_of_annotations:
            Int. Number of annotation vectors for each training sequence.
        :param annotation_size:
            Int. Dimension of each annotation vector.
        :param hidden_state_dimension:
            Number of hidden units in LSTM used for attention model.
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.prediction_classes = prediction_classes
        self.number_of_annotations = number_of_annotations
        self.annotation_size = annotation_size
        self.hidden_state_dimension = hidden_state_dimension
