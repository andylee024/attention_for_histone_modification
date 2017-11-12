class AttentionTrainingExample(object):
    """Training example for attention models."""

    def __init__(self, sequence, label, annotation):
        """Initialize training example.

        :param sequence:
            Numpy array representing genetic sequence.
        :param label:
            Numpy array representing multi-class prediction label.
        :param annotation:
            Numpy array representing annotation vector corresponding to sequence.
        """
        self.sequence = sequence
        self.label = label
        self.annotation = annotation

