#
# Attention for Histone Modification
#

import collections
from itertools import chain

from attention_for_histone_modification.libs.preprocessing.abstract_dataset import AbstractDataset
from attention_for_histone_modification.libs.preprocessing.utilities import load_pickle_object

class ShardedAttentionDataset(AbstractDataset):
    """A sharded attention dataset that satisfies dataset API."""

    def __init__(self, index_to_dataset):
        """Initialize sharded attention dataset.

        :param index_to_dataset:
            Python dictionary where (key, value) pair is training example index and path to 
            attention dataset, respectively.
        """
        self.index_to_dataset = index_to_dataset
        self.total_examples = len(index_to_dataset)

    def get_training_example(self, index):
        """Get training example for query index.
        
        :param index: query index.
        :return: training example for query index.
        """
        return _deserialize_training_example_from_dataset(self.index_to_dataset[index], index)

    def get_training_examples(self, indices):
        """Get training examples corresponding to supplied indices.

        :param indices: Indices corresponding to query (numpy array or python list).
        :return: List of 2-tuples of the form (index, training_example). 
        """
        #_validate_indices(indices, self.index_to_dataset)
        dataset_to_indices_map = _get_dataset_to_indices_list(indices, self.index_to_dataset)
        return list(chain.from_iterable(_deserialize_training_examples_from_dataset(dataset_path, indices) 
                                        for (dataset_path, indices) in dataset_to_indices_map.iteritems()))


class AttentionDatasetInfo(object):
    """A struct that caches information about an attention dataset.
    
    Used to optimize performance of ShardedAttentionDataset.
    """

    def __init__(self, dataset_path, indices):
        """Populate information.

        :param dataset_path:
            Absolute path of attention dataset corresponding to dataset info object.
        :param indices:
            Indices of attention dataset with respect to all samples in all datasets.
        """
        self.dataset_path = dataset_path
        self.indices = indices

    @property
    def number_of_training_examples(self):
        return len(self.indices)

    @property
    def index_to_dataset(self):
        """Return a mapping from training example index to the dataset path."""
        return {index: self.dataset_path for index in self.indices}

# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------

#def _validate_indices(indices, index_to_dataset):
#    """Validate indices are contained in dataset.
#    
#    :param indices: list of indices to query
#    :param index_to_dataset: dictionary mapping index to relevant dataset path
#    """
#    index_not_found = not all((index in index_to_dataset for index in indices))
#    if index_not_found:
#        raise KeyError("One of the supplied indices is not found in the dataset.")


def _get_dataset_to_indices_list(indices, index_to_dataset):
    """Return a dictionary that maps dataset path to list of relevant indices.

    The dictionary contains (key, value) pairs such that the key is the path
    to the dataset and the value is a list of relevant supplied indices corresponding to
    the dataset.

    :param indices: list of indices to query
    :param index_to_dataset: dictionary mapping index to relevant dataset path
    :return: map from dataset path to indices.
    """
    d = collections.defaultdict(list)
    for index in indices:
        dataset_path = index_to_dataset[index]
        d[dataset_path].append(index)
    return d


def _deserialize_training_example_from_dataset(dataset_path, index):
    """Deserialize training example from dataset."""
    dataset = load_pickle_object(dataset_path)
    return dataset.get_training_example(index)


def _deserialize_training_examples_from_dataset(dataset_path, indices):
    """Deserialize dataset and retrieve training examples.

    :param dataset_path:
        Path to pickled dataset to deserialize.
    :param indices:
        Indices corresponding to training examples to deserialize.
    :return:
        List of 2-tuples of the form (index, training_example). 
    """
    dataset = load_pickle_object(dataset_path)
    return dataset.get_training_examples(indices)
