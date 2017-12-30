import numpy as np
import os
import tensorflow as tf


class tf_dataset_wrapper(object):
    """A wrapper for tensorflow datasets to make training more convenient."""

    def __init__(self, config=None):
        """Initialize dataset."""
        self._training_examples = _get_examples_from_config(config)
        self._parse_function = _get_parse_function_from_config(config)

        self.input_examples_op = _get_input_filenames_op()
        self._iterator = None


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
        tf_dataset = tf.data.TFRecordDataset(self.input_filenames_op)
        tf_dataset = tf_dataset.prefetch(buffer_size)
        tf_dataset = tf_dataset.map(_parse_attention_example, num_parallel_calls=6)
        tf_dataset = tf_dataset.batch(self.batch_size)
        self._iterator = tf.dataset.make_initializable_iterator

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


def _get_input_examples_op():
    """Return input examples op associated with dataset.

    The input examples op is a placeholder that holds a list of serialized objects characterizing the dataset examples.
    The current implementation is based on tfrecords, but in general these can be any type of files provided the 
    files can be parsed.
    """
    with tf.variable_scope('input_filenames_op', reuse=False):
        return tf.placeholder(tf.string, shape=[None])


def _get_examples_from_config(config):
    """Return list of .tfrecord files associated with a directory.
   
    :param directory: directory containing tf records.
    :return: list of paths corresponding to tf records.
    """
    #TODO: handle this constant correctly
    TRAINING_EXAMPLE_DIRECTORY = "/Users/andy/Projects/biology/research/komorebi/data/attention_validation_tf_dataset"
    directory = TRAINING_EXAMPLE_DIRECTORY
    return [os.path.join(directory, tf_record) for tf_record in os.listdir(directory)]


def _parse_attention_example(tf_example):
    """Parse tensorflow example type specifically assumed to be attention type.
    
    :param tf_example: example type from tensorflow
    :return: dictionary of tensors with the following attributes.
        
        'sequence'      : sequence tensor (1000, 4)
        'label'         : label tensor (919,)
        'annotation'    : annotation tensor (75, 320)
    """
    # TODO: figure out a better place for parse functions
    # TODO: figure out a cleaner way to handle these constants 
    # TODO: use named-tuple return type rather than dictionary
    SEQUENCE_SHAPE = (1000, 4)
    ANNOTATION_SHAPE = (75, 320)

    # specify features in attention example  
    features_map = {
        'sequence_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
        'annotation_raw': tf.FixedLenFeature([], tf.string)}

    # parse tf example for internal tensors
    parsed_example = tf.parse_single_example(tf_example, features_map)
    sequence_raw = tf.decode_raw(parsed_example['sequence_raw'], tf.uint8)
    label_raw = tf.decode_raw(parsed_example['label_raw'], tf.uint8)
    annotation_raw = tf.decode_raw(parsed_example['annotation_raw'], tf.float32)

    # parsed tensors are flat so reshape if needed
    sequence = tf.reshape(sequence_raw, SEQUENCE_SHAPE)
    label = label_raw
    annotation = tf.reshape(annotation_raw, ANNOTATION_SHAPE)

    return {'sequence': sequence, 'label': label, 'annotation': annotation}

