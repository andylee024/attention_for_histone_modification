#
# Attention for Histone Modification
#
import collections
from copy import deepcopy

from attention_for_histone_modification.libs.preprocessing.attention_dataset import (
        AttentionDataset, AttentionDatasetConfig)
from attention_for_histone_modification.libs.preprocessing.attention_training_example import AttentionTrainingExample
from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.utilities import ensure_samples_match, partition_indices


"""Light-weight struct for carrying partitioned data.

    Attributes:
        indices     : list of ints representing indices into original array on which subarrays are absed.
        sequences   : numpy array representing sequences
        labels      : numpy array containing labels
"""
AttentionPartition = collections.namedtuple(typename='AttentionPartition',
                                            field_names=['indices', 'sequences', 'labels', 'annotations'])


def partition_and_annotate_data(sequences, labels, extractor, partition_size=1000):
    """Partition arrays into smaller subarrays.

    Partition the sequences and labels into subarrays with number of samples equaling partition_size
    and extract corresponding annotations for each subarray pair. The output is a generator yielding
    attention partition structs containing annotations.

    :param sequences:
        Numpy array containing sequence data of shape (number_examples, sequence_length, vocabulary_size)
    :param labels:
        Numpy array containing labels of shape (number_examples, label_dimension)
    :param extractor:
        Extractor object for obtaining annotations.
    :param partition_size:
        Specifies size of each subarray that divides sequences and labels.
    :return:
        2-tuple (iterator to data_partition structs, total number of partitions)
    """
    number_of_samples = ensure_samples_match(sequences, labels)
    index_partitions, total_partitions = partition_indices(number_of_samples, partition_size)

    return (AttentionPartition(indices=index_partition,
                               sequences=sequences[index_partition],
                               labels=labels[index_partition],
                               annotations=extractor.extract_annotation_batch(sequences[index_partition]))
            for index_partition in index_partitions), total_partitions


def create_dataset_from_attention_partition(base_config, attention_partition):
    """Create attention dataset from partition dataset.

    Extract training examples from attention partition struct and update dataset config accordingly.

    :param config: base config used to generate full dataset.
    :param attention_partition: struct containing data needed to generate training examples.
    :return: AttentionDataset type.
    """
    return AttentionDataset(
        config=convert_to_partition_dataset_config(base_config, attention_partition),
        training_examples=generate_training_examples_from_attention_partition(attention_partition))


def generate_training_examples_from_attention_partition(attention_partition):
    """Create list of training examples from attention_partition struct."""
    return [AttentionTrainingExample(sequence=s, label=l, annotation=a)
            for (s, l, a) in zip(attention_partition.sequences,
                                 attention_partition.labels,
                                 attention_partition.annotations)]


def convert_to_partition_dataset_config(base_config, attention_partition):
    """Convert base dataset config to config specific to attention partition.

    The difference between the base_config and the attention_partition config is that
    the attention_partition config contains different start and end indices based on the partitioning.

    :param base_config: Base dataset config used to generate full dataset.
    :param attention_partition: Struct containing partitioned data.
    :return: Dataset config that matches attention partition.
    """
    updated_config = deepcopy(base_config)
    updated_config.dataset_name = "{base_dataset_name}_{start}_{end}".format(base_dataset_name=base_config.dataset_name,
                                                                             start=attention_partition.indices[0],
                                                                             end=attention_partition.indices[-1])
    updated_config.indices = attention_partition.indices
    return updated_config
