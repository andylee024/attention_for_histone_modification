# 
# Attention for Histone Modification
# 


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


class AttentionDataset(object):
    """Dataset for attention models.
    
    Presently, the dataset is just a list of training examples. 
    In the future, we may consider adding configuration properties to the dataset.
    """

    def __init__(self, training_examples):
        """Initialize dataset."""
        assert all((isinstance(te, AttentionTrainingExample) for te in training_examples))
        self.training_examples = training_examples

