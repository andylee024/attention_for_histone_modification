#
# Attention for Histone Modification
#

import datetime
import json
import os


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
    """Dataset for attention models."""

    def __init__(self, config, training_examples):
        """Initialize attention dataset.

        :param config: attention dataset config containing information about dataset.
        :param training_examples: List of generated training examples.
        """
        assert isinstance(config, AttentionDatasetConfig)
        assert all((isinstance(te, AttentionTrainingExample) for te in training_examples))

        self.config = config
        self.training_examples = training_examples

    def get_training_examples(self, indices):
        """Get training examples associated with indices.

        Note that supplied indices must be contained in indices found in attention config.
       
        :param indices: Indices corresponding to training examples to query.
        :return: List of 2-tuples of the form (index, training_example). 
        """
        _validate_indices(indices, self.config)
        normalized_indices = [idx - self.config.indices[0] for idx in indices]
        return [(idx, self.training_examples[query_idx]) for (idx, query_idx) in zip(indices, normalized_indices)]


class AttentionDatasetConfig(object):
    """Config object of attention dataset."""

    def __init__(self, 
                 dataset_name,
                 sequence_data, 
                 label_data,
                 indices,
                 model_name,
                 model_weights,
                 model_layer):
        """Initialize dataset config by populating fields.

        Attributes:
            dataset_name  : name of attention dataset
            sequence_data : path to .npy file containing sequences
            label_data    : path to .npy file containing labels
            indices       : indices corresponding to sequence and label samples associated with dataset
            model_name    : name of model (only DANQ right now)
            model_weights : path to weights of model
            model_layer   : name of model layer to extract annotations
            timestamp     : current timestamp

        """
        self.dataset_name = dataset_name
        self.sequence_data = sequence_data
        self.label_data = label_data
        self.indices = indices
        self.model_name = model_name
        self.model_weights = model_weights
        self.model_layer = model_layer
        self.timestamp = str(datetime.datetime.now())


def _validate_indices(indices, attention_config):
    """Check that supplied indices are all contained in config.
    
    :param indices: list of indices to check
    :param attention_config: attention dataset configuration object.
    """
    if not all([(idx in attention_config.indices) for idx in indices]):
        raise IndexError("Supplied indices not contained in dataset.")

