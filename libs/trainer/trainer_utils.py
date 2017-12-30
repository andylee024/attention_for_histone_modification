"""Utilities for trainer."""

from komorebi.libs.dataset.types.abstract_dataset import AbstractDataset
from komorebi.libs.utilities.array_utils import get_shuffled_indices, partition_indices

def batch_data(dataset, batch_size=100):
    """Create batches of training example for one epoch.

    :param dataset: object of type AbstractDataset.
    :param batch_size: size of each batch.
    :return: 2-tuple
        1st element: generator yielding lists of training examples of size "batch_size"
        2nd element: total batches 
    """
    assert isinstance(dataset, AbstractDataset)
    index_partitions, total_batches = partition_indices(get_shuffled_indices(dataset.total_examples), partition_size=batch_size)
    return (dataset.get_training_examples(ip) for ip in index_partitions), total_batches


def get_epoch_iterator(tf_dataset_wrapper, sess):
    """Get iterator for epoch training.

    Shuffle dataset examples and return corresponding new iterator to shuffled examples.
    
    :param tf_dataset_wrapper: dataset wrapper object
    :param sess: tf session
    """
    shuffled_examples = tf_dataset_wrapper.get_shuffled_training_files()
    sess.run(tf_dataset_wrapper.iterator.initializer, 
             feed_dict={tf_dataset_wrapper.input_examples_op: shuffled_examples})
    return tf_dataset_wrapper.iterator


def compute_number_of_batches(tf_dataset_wrapper, batch_size):
    """Compute number of partitions for dataset given batch_size.
    
    :param tf_dataset_wrapper: dataset on which we are iterating over
    :param batch_size: size of each iteration batch
    :return: total number of partitions
    """
    _, total_batches = partition_indices(indices=range(tf_dataset_wrapper.number_of_examples), 
                                         partition_size=batch_size)
    return total_batches

