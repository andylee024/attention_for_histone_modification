# 
# Attention for Histone Modification
#

"""Utilities for dataset generation."""

import collections
import numpy as np

from attention_for_histone_modification.libs.utilities.validate import ensure_samples_match

"""
Light-weight struct for carrying partitioned data.

    Attributes:
        indices     : list of ints representing indices into original array on which subarrays are absed.
        sequences   : numpy array representing sequences
        labels      : numpy array containing labels
"""
data_partition = collections.namedtuple(typename='data_partition', field_names=['indices', 'sequences', 'labels'])


def get_partition_data_stream(sequences, labels, partition_size=1000):
    """Partition arrays into smaller subarrays. 

    A list of data_partition structs are returned. Each data_partition contains subarrays that have samples
    equal to the partition_size, up until the last data_partition which contains the remaining samples of the
    full input arrays.
    
    :param array: Numpy array to split up.
    :param partition_size: Maximum size of each sub-array.
    :return: Generator of data_partition structs.
    """
    number_of_samples = ensure_samples_match(sequences, labels)
    return (data_partition(indices=index_partition, 
                           sequences=sequences[index_partition], 
                           labels=labels[index_partition]) 
                           for index_partition in _partition_indices(number_of_samples, partition_size))


def _partition_indices(number_of_samples, partition_size):
    """Given number of samples and partition size, generate list of partition indices.
    
    E.G. 
        number_of_samples = 10
        partition_size = 3
        return: [ [0,1,2], [3,4,5], [6,7,8], [9] ]
    """
    assert partition_size <= number_of_samples

    # index by 1, otherwise split operation yields a list with the first element as an empty list.
    cutoffs = np.arange(start=0, stop=number_of_samples, step=partition_size)[1:]
    indices = np.arange(number_of_samples)
    return np.split(indices, cutoffs)


