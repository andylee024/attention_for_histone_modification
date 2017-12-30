"""Utilities for trainer."""
from komorebi.libs.utilities.array_utils import partition_indices

def get_data_stream_for_epoch(dataset, sess):
    """Get data stream for epoch training.

    Shuffle dataset examples and return corresponding tf iterator op to shuffled examples.

    Note we return the iterator op rather than the iterator because each call to get_next()
    adds an additional op on the tensorflow graph. Returning the op, calls the operation
    only once per epoch.
    
    :param dataset: tf_dataset_wrapper object
    :param sess: tensorflow session
    :return: tensorflow op for streamining data
    """
    shuffled_examples = dataset.get_shuffled_training_files()
    sess.run(dataset.iterator.initializer, 
             feed_dict={dataset.input_examples_op: shuffled_examples})
    shuffled_iterator = dataset.iterator
    data_stream_op = shuffled_iterator.get_next()
    return data_stream_op


def compute_number_of_batches(dataset, batch_size):
    """Compute number of partitions for dataset given batch_size.
    
    :param dataset: dataset on which we are iterating over
    :param batch_size: size of each iteration batch
    :return: total number of partitions
    """
    _, total_batches = partition_indices(indices=range(dataset.number_of_examples), 
                                         partition_size=batch_size)
    return total_batches

