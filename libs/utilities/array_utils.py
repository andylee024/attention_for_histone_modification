import numpy as np

"""Utilities for numpy arrays."""

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


def partition_indices(indices, partition_size):
    """Given indices and partition size, partition indices into smaller chunks.

    Note that the relative ordering of the supplied indices are preserved 
    by this function.

    :param indices: list or numpy array of indices to partition.
    :param partition_size: maximum size of each partition

    E.G.
        indices = [0, 1, ... , 9]
        partition_size = 3
        return: [ [0,1,2], [3,4,5], [6,7,8], [9] ]
    """
    assert partition_size <= len(indices)

    # need to index by 1, otherwise split operation yields a list empty first element
    cutoffs = np.arange(start=0, stop=len(indices), step=partition_size)[1:]
    partitions = np.split(indices, cutoffs)
    return partitions, len(partitions)


def get_shuffled_indices(number_of_samples):
    """Return shuffled indices given number of samples."""
    indices = np.arange(number_of_samples)
    np.random.shuffle(indices)
    return indices

