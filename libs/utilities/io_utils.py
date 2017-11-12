import os
import pickle
import shutil

"""Utilities for i/o operations."""

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

