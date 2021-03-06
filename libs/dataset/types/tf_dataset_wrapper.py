import numpy as np
import os
import tensorflow as tf

from komorebi.libs.dataset.parsing.tf_example_parsers import parse_attention_example
from komorebi.libs.dataset.types.dataset_config import DatasetConfiguration


class tf_dataset_wrapper(object):
    """A wrapper for tensorflow datasets to make training more convenient."""

    def __init__(self, config):
        """Initialize dataset."""
        assert isinstance(config, DatasetConfiguration)
        
        self._name = config.dataset_name
        self._training_examples = _get_examples_from_directory(config.examples_directory)
        self._parse_function  = parse_attention_example

        self._iterator = None
        self.input_examples_op = _get_input_examples_op()

    def get_shuffled_training_files(self):
        """Return shuffled list of training example files."""
        return np.random.permutation(self._training_examples)

    def build_input_pipeline_iterator(self, batch_size, buffer_size, parallel_calls):
        """Build and return dataset iterator object with specified input processing parameters.
        
        :param batch_size: batch_size of returned examples
        :param buffer_size: number of examples to buffer in memory 
        :param parallel_calls: number of threads to spin off for processing input files
        :return: iterator object
        """
        tf_dataset = tf.data.TFRecordDataset(self.input_examples_op)
        tf_dataset = _build_specialized_dataset(tf_dataset=tf_dataset, 
                                                parse_function=self._parse_function,
                                                batch_size=batch_size,
                                                buffer_size=buffer_size,
                                                parallel_calls=parallel_calls)
        self._iterator = tf_dataset.make_initializable_iterator()

    def build_one_shot_iterator(self):
        """Build a one-shot iterator for dataset."""
        tf_dataset = tf.data.TFRecordDataset(self.input_examples_op)
        self._iterator = tf_dataset.make_one_shot_iterator()

    @property
    def number_of_examples(self):
        """Return number of training examples associated with dataset."""
        return len(self._training_examples)


    @property
    def iterator(self):
        """Return iterator object associated with dataset."""
        if self._iterator is None:
            raise RuntimeError("Iterator is not initialized for dataset!")
        return self._iterator


def _build_specialized_dataset(tf_dataset, parse_function, batch_size, buffer_size, parallel_calls):
    """Build and return tensorflow dataset with specified configurations for iterating.
    
    :param tf_dataset: base tensorflow dataset to transform
    :param parse_function: function used to parse dataset examples 
    :param batch_size: batch_size of returned examples
    :param buffer_size: number of examples to buffer in memory 
    :param parallel_calls: number of threads to spin off for processing input files
    :return: dataset object
    """
    tf_dataset = tf_dataset.prefetch(buffer_size)
    tf_dataset = tf_dataset.map(map_func=parse_function, num_parallel_calls=parallel_calls)
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset


def _get_input_examples_op():
    """Return input examples op associated with dataset.

    The input examples op is a placeholder that holds a list of serialized objects characterizing the dataset examples.
    The current implementation is based on tfrecords, but in general these can be any type of files provided the 
    files can be parsed.
    """
    with tf.variable_scope('input_examples_op', reuse=False):
        return tf.placeholder(tf.string, shape=[None])


def _get_examples_from_directory(examples_directory):
    """Return list of .tfrecord files associated with a directory.
   
    :param examples_directory: directory containing tf records
    :return: list of paths corresponding to tf records
    """
    if os.path.isdir(examples_directory):
        return [os.path.join(examples_directory, tf_record) for tf_record in os.listdir(examples_directory)]
    raise IOError("Directory {} does not exist!".format(examples_directory))

