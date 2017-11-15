#
# Unit tests for extractor
#

import numpy as np
import unittest

from komorebi.libs.dataset.preprocessing.extractor import (
        AnnotationExtractor, get_trained_danq_model)

# Configuration variables
DANQ_WEIGHTS_FILE = '/Users/andy/Projects/biology/research/komorebi/data/danq_weights.hdf5'
LAYER_NAME = 'dense_1'
ANNOTATION_VECTOR_SIZE = 925


def create_dummy_batch_sequence(
        batch_size=100, sequence_length=1000, vocabulary_size=4):
    """Creates batch of training example sequences for testing.

    :param batch_size: Int. Size of batch.
    :param sequence_length: Length of sequences in prediction problem.
    :param vocabulary_size: Size of sequence vocabulary.
    :return: Numpy array of dimension (batch_size, sequence_length, vocabulary_size).
    """
    return np.zeros(shape=(batch_size, sequence_length, vocabulary_size))


class TestAnnotationExtractor(unittest.TestCase):
    """Tests for extracting annotation vectors."""

    @classmethod
    def setUpClass(cls):
        """Initialize  extractor for tests."""
        cls.extractor = AnnotationExtractor(
            model=get_trained_danq_model(DANQ_WEIGHTS_FILE),
            layer_name=LAYER_NAME)

    def test_annotation_extraction(self):
        """Test annotation extraction for single sequence."""
        dummy_training_sequence = create_dummy_batch_sequence(batch_size=1)
        annotation_vector = self.extractor.extract_annotation(
            dummy_training_sequence)

        expected_annotation_vector_size = ANNOTATION_VECTOR_SIZE
        self.assertEqual(
            annotation_vector.size,
            expected_annotation_vector_size)

    def test_batch_annotation_extraction(self):
        """Test annotation extraction for single sequence."""
        batch_size = 100
        dummy_training_sequence_batch = create_dummy_batch_sequence(
            batch_size=batch_size)
        annotation_vectors = self.extractor.extract_annotation_batch(
            dummy_training_sequence_batch)

        expected_shape = (batch_size, 925)
        self.assertSequenceEqual(annotation_vectors.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
