import numpy as np
import unittest

from komorebi.libs.utilities.array_utils import ensure_samples_match, partition_indices

class TestArrayUtils(unittest.TestCase):
    """Tests for array utils."""

    def test_ensure_samples_match_raises(self):
        """Ensure samples raises for mismatch in array sizes."""
        self.assertRaises(ValueError, ensure_samples_match, np.arange(100), np.arange(4))

    def test_ensure_samples_returns_correct_samples(self):
        """Ensure samples should return number of samples for matching array sizes along axis 0."""
        expected_samples = 100
        dummy_array = np.arange(expected_samples)
        self.assertEqual(ensure_samples_match(dummy_array, dummy_array), expected_samples)

    def test_partition_indices_returns_correct_indices(self):
        number_of_samples = 3
        partition_size = 2

        # [0, 1, 2] -> [ [0, 1], [2] ]
        expected_indices = [np.array([0, 1]), np.array([2])]
        actual_indices, _ = partition_indices(np.arange(number_of_samples), partition_size)

        self.assertEqual(len(actual_indices), len(expected_indices))
        for (ai, ei) in zip(actual_indices, expected_indices):
            np.testing.assert_array_equal(ai, ei)

if __name__ == '__main__':
    unittest.main()
