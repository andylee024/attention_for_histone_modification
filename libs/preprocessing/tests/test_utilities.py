#
# Unit tests for dataset utilities
#

import numpy as np
import unittest

from attention_for_histone_modification.libs.preprocessing.utilities import get_partition_data_stream


class TestUtilities(unittest.TestCase):
    """Tests for extracting annotation vectors."""

    def test_partition_data_throws_for_incorrect_sample_size(self):
        sequences = np.arange(100)
        labels = np.arange(4)
        self.assertRaises(ValueError, get_partition_data_stream, sequences, labels)

    def test_partion_data_returns_correct_partitions(self):
        """Test that correct partitions are returned."""
        dummy_array = np.array([[0, 0], [1, 1], [2, 2]]) 

        # [0, 1, 2] -> [ [0, 1], [2] ]
        expected_indices = [np.array([0, 1]), np.array([2])] 

        # [ [0,0], [1,1], [2,2] ] -> [ [ [0,0], [1,1] ], [ [2, 2] ] ]
        expected_data = [np.array([[0, 0], [1, 1]]), np.array([[2, 2]])] 

        partition_data_stream = get_partition_data_stream(sequences=dummy_array, labels=dummy_array, partition_size=2)
        for idx, pd in enumerate(partition_data_stream):
            np.testing.assert_array_equal(pd.indices, expected_indices[idx])
            np.testing.assert_array_equal(pd.labels, expected_data[idx])
            np.testing.assert_array_equal(pd.sequences, expected_data[idx])


if __name__ == '__main__':
    unittest.main()
