# 
# Attention for Histone Modification
# 

class AttentionTrainingExample(object):
    """Training example for attention models."""

    def __init__(self, sequence, label, annotation_vector):
        """Initialize training example.

        :param sequence:
            Numpy array representing genetic sequence.
        :param label:
            Numpy array representing multi-class prediction label.
        :param annotation_vector:
            Numpy array representing annotation vector corresponding to sequence.
        """
        self.sequence = sequence
        self.label = label
        self.annotation_vector = annotation_vector


class AttentionDataset(object):
    """Dataset for attention models.
    
    Presently, the dataset is just a list of training examples. 
    In the future, we may consider adding configuration properties to the dataset.
    """

    def __init__(self, training_examples):
        """Initialize dataset."""
        assert all((isinstance(te, AttentionTrainingExample) for te in training_examples))
        self._training_examples = training_examples

