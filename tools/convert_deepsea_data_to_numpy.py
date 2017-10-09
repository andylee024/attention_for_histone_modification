# 
# Attention for Histone Modification
#
# A command line tool for converting deepsea data into numpy arrays.
#

import argparse 
import collections
import h5py
import numpy as np
import os
import scipy.io
import sys

# Specify dimensions for converting matrix shape.
TRANSPOSE_AXES_X = {'train': (2, 0, 1), 'test': (0, 2, 1), 'valid': (0, 2, 1)}
TRANSPOSE_AXES_Y = {'train': (1, 0), 'test': (0, 1), 'valid': (0, 1)}

# data type for holding ml data with split label
ml_datum = collections.namedtuple(typename='ml_datum', field_names=['x', 'y', 'split'])


def _validate_paths(data_paths):
    """Validate that data paths exist in system.

    :param data_paths:
        Python dictionary, where key is split name and value is data path.
    """
    for (split, path) in data_paths.iteritems():
        if not os.path.isfile(path):
            raise OSError("Data file {} does not exist!".format(path))


def _get_valid_data_paths(args):
    """Return all valid data paths for deepsea data extraction.
    
    :return 
        Python dictionary, where key is split name and value is data path.
    """
    all_data_paths = {'train': args.train, 'test': args.test, 'valid': args.validation}
    filtered_data_paths = {split: path for (split, path) in all_data_paths.iteritems() if path is not None}
   
    _validate_paths(filtered_data_paths)
    return filtered_data_paths


def _get_raw_deepsea_data(split, path):
    """Parse data paths and get raw deepsea data.
    
    :param path: Path to raw (.mat) data
    :param split: Str. Type of split in data.
    :return: ml_datum type containing x, y, split information
    """
    x_data_key = "{}xdata".format(split)
    y_data_key = "{}data".format(split)

    if split == "train":
        data = h5py.File(path)
    elif split == "test":
        data = scipy.io.loadmat(path)
    elif split == "valid":
        data = scipy.io.loadmat(path)
    else:
        raise NotImplementedError("split type not recognized!")

    return ml_datum(x=data[x_data_key], y=data[y_data_key], split=split)


def _process_deepsea_ml_datum(mld):
    """Convert dimensions of ML datums.
    
    The dimensions in the raw deepsea data is not consistent. 
    This function standardizes the dimensions of each numpy matrix from deepsea data.
    
    :param mld: original ml_datum representing raw deepsea data
    :return: ml_datum with containing matrices with converted shapes
    """
    processed_x = np.transpose(mld.x, TRANSPOSE_AXES_X[mld.split])
    processed_y = np.transpose(mld.y, TRANSPOSE_AXES_Y[mld.split])
    print "converted {0}_X.shape {1} -> {2}".format(mld.split, mld.x.shape, processed_x.shape)
    print "converted {0}_Y.shape {1} -> {2}".format(mld.split, mld.y.shape, processed_y.shape)
    return ml_datum(x=processed_x, y=processed_y, split=mld.split)


def _save_ml_datum(mld, directory):
    """Save numpy matrices of ml datum."""
    destination_x = os.path.join(directory, "deepsea_{}_x.npy".format(mld.split))
    destination_y = os.path.join(directory, "deepsea_{}_y.npy".format(mld.split))

    np.save(destination_x, mld.x)
    print "saved...{}".format(destination_x)

    np.save(destination_y, mld.y)
    print "saved...{}".format(destination_y)


def main(args):
    data_paths = _get_valid_data_paths(args)
    deepsea_ml_data = (_get_raw_deepsea_data(split, path) for (split, path) in data_paths.iteritems())
    ml_data = (_process_deepsea_ml_datum(mld) for mld in deepsea_ml_data)
    for mld in ml_data:
        _save_ml_datum(mld, args.directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for extracting data from deepsea dataset.")
    parser.add_argument("--train", type=str, help="Path to vaildation .mat file.")
    parser.add_argument("--test", type=str, help="Path to vaildation .mat file.")
    parser.add_argument("--validation", type=str, help="Path to vaildation .mat file.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to output directory.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
