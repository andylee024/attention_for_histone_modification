#
# Attention for Histone Modification
#

import numpy as np


def ensure_samples_match(*arrays):
    """Ensure that input arrays have same number of samples.

    :param *arrays:
        Variable length arguments of arrays.
    :return: Int. Number of samples of each array
    :exeception: ValueError.
    """
    number_of_samples = len(arrays[0])
    if all((len(a) == number_of_samples for a in arrays)):
        return number_of_samples
    else:
        raise ValueError("Arrays do not same number of samples!")


def partition_indices(number_of_samples, partition_size):
    """Given number of samples and partition size, generate list of partition indices.

    E.G.
        number_of_samples = 10
        partition_size = 3
        return: [ [0,1,2], [3,4,5], [6,7,8], [9] ]
    """
    assert partition_size <= number_of_samples

    # index by 1, otherwise split operation yields a list with the first
    # element as an empty list.
    cutoffs = np.arange(start=0, stop=number_of_samples, step=partition_size)[1:]
    indices = np.arange(number_of_samples)
    partitions = np.split(indices, cutoffs)
    return partitions, len(partitions)
