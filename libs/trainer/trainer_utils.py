"""Utilities for trainer."""

import numpy as np
from tqdm import tqdm

from komorebi.libs.utilities.array_utils import partition_indices

# Struct storing relevant dataset statistics
dataset_statistics = collections.namedtuple(typename="dataset_statistics", 
                                           field_names=['positive_samples', 'negative_samples'])


def infer_positive_upweight_parameter(tf_dataset, task_id, sess):
    """Infer positive upweight parameter for imbalanced datasets.

    In classification settings, the imbalanced dataset issue often arises. In these cases, we would like to weight
    the losses for positive and negative samples accordingly to balance the dataset.

    :param tf_dataset: dataset object
    :param task_id: task for which to compute statistics.
    :param sess: tensorflow session 
    :return: float. upweight parameter for positive samples
    """
    dataset_stats = _get_dataset_statistics_for_task(tf_dataset, task_id, sess)
    return negative_samples / positive_samples
    

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


def _get_dataset_statistics_for_task(tf_dataset, task_id, sess):
    """Get statistics for a particular task for a multitask dataset.

    :param tf_dataset: dataset object
    :param task_id: task for which to compute statistics.
    :param sess: tensorflow session
    :return: 2-tuple (number of positive samples, number of negative samples)
    """
    tf_dataset.build_input_pipeline_iterator(batch_size=1, buffer_size=1000, parallel_calls=6)
    total_examples = tf_dataset.number_of_examples
    data_stream_op = get_data_stream_for_epoch(tf_dataset, sess)

    training_examples = (sess.run(data_stream_op) for _ in xrange(tf_dataset.number_of_examples))
    multitask_labels = (np.ravel(te['label']) for te in training_examples)
    single_task_labels = [mt_label[task_id] for mt_label in tqdm(multitask_labels, 
                                                                 desc="computing dataset statistics", 
                                                                 total=total_examples)]
    
    positive_samples = sum(single_task_labels)
    return dataset_statistics(positive_samples=positive_samples,
                              negative_samples=total_examples-positive_samples)
