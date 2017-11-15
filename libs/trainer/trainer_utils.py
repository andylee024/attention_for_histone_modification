"""Utilities for trainer."""

from komorebi.libs.dataset.types.abstract_dataset import AbstractDataset
from komorebi.libs.utilities.array_utils import get_shuffled_indices, partition_indices

def batch_data(dataset, batch_size=100):
    """Create batches of training example for one epoch.

    :param dataset: object of type AbstractDataset.
    :param batch_size: size of each batch.
    :return: generator yielding lists of training examples of size "batch_size"
    """
    assert isinstance(dataset, AbstractDataset)
    index_partitions, _ = partition_indices(get_shuffled_indices(dataset.total_examples), partition_size=batch_size)
    return (dataset.get_training_examples(ip) for ip in index_partitions)

