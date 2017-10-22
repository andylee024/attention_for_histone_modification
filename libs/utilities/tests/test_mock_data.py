#
# Unit tests for dataset utilities
#

import numpy as np
import unittest

from attention_for_histone_modification.libs.utilities.mock_data import *


# fixed constants for testing
sequence_length = 10
vocabulary_size = 20
prediction_classes = 30
number_of_annotations = 40
annotation_dimension = 50
batch_size = 60


class TestMockData(unittest.TestCase):
    """Tests shapes for mock data."""

    def test_create_dummy_sequence(self):
        dummy_sequence = create_dummy_sequence(sequence_length, vocabulary_size)
        expected_shape = (sequence_length, vocabulary_size)
        self.assertSequenceEqual(dummy_sequence.shape, expected_shape)

    def test_create_dummy_label(self):
        dummy_label = create_dummy_label(prediction_classes)
        expected_size = prediction_classes
        self.assertEqual(dummy_label.size, expected_size)

    def test_create_dummy_annotation(self):
        dummy_annotation = create_dummy_annotation(number_of_annotations, annotation_dimension)
        expected_shape = (number_of_annotations, annotation_dimension)
        self.assertSequenceEqual(dummy_annotation.shape, expected_shape)

    def test_create_dummy_sequence_batch(self):
        dummy_sequence_batch = create_dummy_sequence_batch(
            sequence_length, vocabulary_size, batch_size)
        expected_shape = (batch_size, sequence_length, vocabulary_size)
        self.assertSequenceEqual(dummy_sequence_batch.shape, expected_shape)

    def test_create_dummy_label_batch(self):
        dummy_label_batch = create_dummy_label_batch(prediction_classes, batch_size)
        expected_shape = (batch_size, prediction_classes)
        self.assertSequenceEqual(dummy_label_batch.shape, expected_shape)

    def test_create_dummy_annotation_batch(self):
        dummy_annotation_batch = create_dummy_annotation_batch(
            number_of_annotations, annotation_dimension, batch_size)
        expected_shape = (batch_size, number_of_annotations, annotation_dimension)
        self.assertSequenceEqual(dummy_annotation_batch.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
