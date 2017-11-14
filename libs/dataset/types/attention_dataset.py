
import datetime

from attention_for_histone_modification.libs.dataset.types.abstract_dataset import AbstractDataset
from attention_for_histone_modification.libs.dataset.types.attention_training_example import AttentionTrainingExample


"""
An implementation of the abstract dataset for attention based models.

|-----------------------------|
|AbstractDataset <<interface>>|
|-----------------------------|
    |
    |-----AttentionDataset <<implementation>>
"""

class AttentionDataset(AbstractDataset):
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

    def _normalize_index(self, index):
        """Normalize index according to indices from config."""
        return index - self.config.indices[0] 

    def get_training_example(self, index):
        """Get training example corresponding to index.
        
        :param index: query index
        :return: training example corresponding to query index.
        """
        return self.training_examples[self._normalize_index(index)]

    def get_training_examples(self, indices):
        """Get training examples associated with indices.

        Note that supplied indices must be contained in indices found in attention config.
       
        :param indices: Indices corresponding to training examples to query.
        :return: List of 2-tuples of the form (index, training_example). 
        """
        _validate_indices(indices, self.config)
        #return [(index, self.get_training_example(index)) for index in indices] (old-implementation)
        return [self.get_training_example(index) for index in indices]


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

