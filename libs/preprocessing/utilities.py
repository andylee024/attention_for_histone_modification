#
# Attention for Histone Modification
#

import numpy as np
import os
import pickle
import shutil

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

# TODO: Change this so that it works for training 
#def partition_indices(indices, partition_size):
#    """Given number of samples and partition size, generate list of partition indices.
#
#    E.G.
#        number_of_samples = 10
#        partition_size = 3
#        return: [ [0,1,2], [3,4,5], [6,7,8], [9] ]
#    """
#    number_of_samples = len(indices)
#    assert partition_size <= number_of_samples
#
#    # index by 1, otherwise split operation yields a list with the first
#    # element as an empty list.
#    cutoffs = np.arange(start=0, stop=number_of_samples, step=partition_size)[1:]
#    partitions = np.split(indices, cutoffs)
#    return partitions, len(partitions)


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


def load_pickle_object(path):
    """Load a pickled object for supplied path."""
    if not os.path.isfile(path):
        raise IOError("Attempted to load object of path {}, but path does not exist!".format(path))
    with open(path, 'r') as f:
        return pickle.load(f)


def write_object_to_disk(obj, path, logger=None):
    """Write object to disk at specified location.
    
    :param obj: object to pickle
    :param path: path to save location
    :param logger: if supplied, then log save status
    """
    if os.path.exists(path):
        raise IOError("Attempted to write object to path {}, but path already exists!".format(path))

    with open(path, 'w') as f:
        pickle.dump(obj, f)

    if logger:
        logger.info("\t Saved {}".format(path))


def copy_data(source, destination, logger=None):
    """Copy source file to destination directory and log information.
    
    :param source: file to copy
    :param destination: destination directory
    :param logger: if supplied, then log copy status
    """
    assert os.path.isfile(source)
    assert os.path.isdir(destination)
    shutil.copy(source, destination)

    if logger:
        logger.info("\t copied {} to {}".format(os.path.basename(source), destination))

def remove_directory(directory, logger=None):
    """Remove specified directory.

    :param drectory: directory to remove
    :param logger: if supplied, then log removal status
    """
    if not os.path.isdir(directory):
        raise IOError("Tried to remove {}, but directory does not exist!".format(directory))

    shutil.rmtree(directory)

    if logger:
        logger.info("\t deleted {}".format(directory))

