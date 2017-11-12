#
# Unit tests for extractor
#

import numpy as np
import unittest

from attention_for_histone_modification.libs.dataset.preprocessing.shard_processing import (
        convert_to_partition_dataset_config, 
        generate_training_examples_from_attention_partition, 
        partition_and_annotate_data)
from attention_for_histone_modification.libs.dataset.preprocessing.extractor import (
        AnnotationExtractor, get_trained_danq_model)
from attention_for_histone_modification.libs.utilities.mock_data import (
        create_dummy_dataset_config, create_dummy_sequence_batch, create_dummy_label_batch)

# Configuration variables
DANQ_WEIGHTS_FILE = '/Users/andy/Projects/biology/research/attention_for_histone_modification/data/danq_weights.hdf5'
LAYER_NAME = 'dense_1'
ANNOTATION_DIMENSION = 925


class TestBatchProcessing(unittest.TestCase):
    """Tests for extracting annotation vectors."""

    @classmethod
    def setUpClass(cls):
        """Initialize data for tests."""

        cls.partition_size = 100
        cls.batch_size = 1001

        # initialize extractor
        cls.extractor = AnnotationExtractor(model=get_trained_danq_model(DANQ_WEIGHTS_FILE), layer_name=LAYER_NAME)

        # initialize sequence data
        cls.sequence_length = 1000
        cls.vocabulary_size = 4
        cls.sequence_data = create_dummy_sequence_batch(sequence_length=cls.sequence_length,
                                                        vocabulary_size=cls.vocabulary_size,
                                                        batch_size=cls.batch_size)

        # initialize label data
        cls.prediction_classes = 3
        cls.label_data = create_dummy_label_batch(prediction_classes=cls.prediction_classes,
                                                  batch_size=cls.batch_size)

        # create partition data
        partition_data_stream, _ = partition_and_annotate_data(sequences=cls.sequence_data,
                                                               labels=cls.label_data,
                                                               extractor=cls.extractor,
                                                               partition_size=cls.partition_size)

        cls.partition_data = list(partition_data_stream)

    def test_partition_and_annotate_data(self):
        """Check that array shapes are correct after partitioning and annotating data."""

        # check intermediate partition
        intermediate_partition_data = self.partition_data[:-1]

        expected_intermediate_indices_size = self.partition_size
        expected_intermediate_sequences_shape = (self.partition_size, self.sequence_length, self.vocabulary_size)
        expected_intermediate_labels_shape = (self.partition_size, self.prediction_classes)

        for pd in intermediate_partition_data:
            self.assertEqual(pd.indices.size, expected_intermediate_indices_size)
            self.assertSequenceEqual(pd.sequences.shape, expected_intermediate_sequences_shape)
            self.assertSequenceEqual(pd.labels.shape, expected_intermediate_labels_shape)

        # check last partition
        last_partition_data = self.partition_data[-1]

        expected_last_indices_size = 1
        expected_last_sequences_shape = (1, self.sequence_length, self.vocabulary_size)
        expected_last_labels_shape = (1, self.prediction_classes)

        self.assertEqual(last_partition_data.indices.size, expected_last_indices_size)
        self.assertSequenceEqual(last_partition_data.sequences.shape, expected_last_sequences_shape)
        self.assertSequenceEqual(last_partition_data.labels.shape, expected_last_labels_shape)

    def test_generate_training_examples_from_partition_data(self):
        """Test training examples generated correctly."""
        single_attention_partition = self.partition_data[0]
        training_examples = generate_training_examples_from_attention_partition(single_attention_partition)

        expected_sequence_shape = (self.sequence_length, self.vocabulary_size)
        expected_label_size = self.prediction_classes
        expected_annotation_size = ANNOTATION_DIMENSION

        self.assertEqual(len(training_examples), self.partition_size)
        for te in training_examples:
            self.assertSequenceEqual(te.sequence.shape, expected_sequence_shape)
            self.assertEqual(te.label.size, expected_label_size)
            self.assertEqual(te.annotation.size, expected_annotation_size)

    def test_convert_to_partition_dataset_config(self):
        """Test dataset config updated correctly according to attention partition."""
        base_config = create_dummy_dataset_config()
        attention_partition = self.partition_data[0]

        partition_config = convert_to_partition_dataset_config(base_config, attention_partition)
        np.testing.assert_array_equal(partition_config.indices, attention_partition.indices)


if __name__ == '__main__':
    unittest.main()
