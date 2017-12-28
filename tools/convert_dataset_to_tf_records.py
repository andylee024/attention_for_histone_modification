import argparse
import logging
import numpy as np
import os
import sys
import tensorflow as tf
from tqdm import tqdm

from komorebi.libs.utilities.array_utils import partition_indices
from komorebi.libs.utilities.io_utils import ensure_directory, load_pickle_object

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PARTITION_SIZE = 1000

def main(args):
    """Convert attention datasets into tf records."""
    _handle_overwrite(args.directory, args.overwrite)
    _handle_directory_creation(args.directory)

    sharded_dataset = load_pickle_object(args.sharded_dataset)
    index_partitions, total_partitions = partition_indices(range(sharded_dataset.total_examples), PARTITION_SIZE)

    for ip in tqdm(index_partitions, total=total_partitions, desc="partitions"):
        training_examples = sharded_dataset.get_training_examples(ip)
        tf_examples = [convert_to_tf_example(training_example) for training_example in training_examples]

        indexed_tf_examples = zip(ip, tf_examples)
        for (index, tf_example) in tqdm(indexed_tf_examples, desc="\t partition progress"):
            _write_tf_example(tf_example=tf_example, path=_get_tfrecord_path(args.directory, index))


def _handle_overwrite(dataset_directory, overwrite_flag=False):
    """Overwrite of specified directory if overwrite flag is set.

    :param dataset_directory: Path to dataset directory.
    :param overwrite: overwrite flag option.
    """
    if overwrite_flag:
        logger.info("--overwrite flag set, Initiating overwrite routine...")
        remove_directory(dataset_directory, logger=logger)


def _handle_directory_creation(directory):
    """Create directory for tf records."""
    logger.info("Initiating directory creation routine...")
    ensure_directory(directory)


def _write_tf_example(tf_example, path):
    """Create tf writer and write tf example to disk.

    :param tf_example: Example type from tensorflow to write
    :param path: path to write tf example.
    """
    if os.path.exists(path):
        raise IOError("Path {} alredy exists!".format(path))
    writer = tf.python_io.TFRecordWriter(path)
    writer.write(tf_example.SerializeToString())
    writer.close()


def _get_tfrecord_path(directory, index):
    """Get path to tf record for a given example.

    :param directory: output directory for tf records
    :param index: index of example in dataset.
    """
    tag = "example_{}.tfrecord".format(index)
    return os.path.join(directory, tag)


def _int64_feature(value):
    """Convert value to int64 feature object in tensorflow for serialization."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Convert value to raw bytes feature object in tensorflow for serialization."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tf_example(training_example):
    """Convert attention training example type to tf example.
    
    For some reason, the features keyword argument needs to be present in Features() object.
    """
    return tf.train.Example(features=tf.train.Features(
        feature={
            'sequence_raw': _bytes_feature(training_example.sequence.tostring()),
            'label_raw': _bytes_feature(training_example.label.tostring()),
            'annotation_raw': _bytes_feature(training_example.annotation.tostring())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for converting sharded dataset to tf records.")
    parser.add_argument("-s", "--sharded-dataset", type=str, required=True, help="Path to sharded dataset.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to output directory for tf records.")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite directory if it exists.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
