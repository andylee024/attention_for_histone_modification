#
# Unit tests for extractor
#
import tempfile
import unittest
import os

from komorebi.libs.dataset.types.sharded_attention_dataset import (
        AttentionDatasetInfo, ShardedAttentionDataset)
from komorebi.libs.utilities.io_utils import remove_directory, write_object_to_disk
from komorebi.libs.dataset.types.tests.unittest_helpers import (
        create_single_example_dataset_with_label)

class TestShardedAttentionDataset(unittest.TestCase):
    """Tests for SharededAttentionDataset."""

    @classmethod
    def setUpClass(cls):
        """Initialize  attention dataset for tests."""

    def setUp(self):
        """Common setup for unit test."""
        self.tmpdir = tempfile.mkdtemp()
       
        # create attention datasets to shard
        dataset_0_path = os.path.join(self.tmpdir, "dataset_0")
        dataset_1_path = os.path.join(self.tmpdir, "dataset_1")
        write_object_to_disk(obj=create_single_example_dataset_with_label(0), path=dataset_0_path)
        write_object_to_disk(obj=create_single_example_dataset_with_label(1), path=dataset_1_path)

        # create sharded attention dataset info
        index_to_dataset = {0: dataset_0_path, 1: dataset_1_path}
        self.sharded_attention_dataset = ShardedAttentionDataset(index_to_dataset)

    def tearDown(self):
        """Clean up for unit test."""
        remove_directory(self.tmpdir)

    def test_get_training_example(self):
        """Test single training example can be queried."""

        # check that key error is raised for incorrect indices
        invalid_index = 3
        self.assertRaises(KeyError, self.sharded_attention_dataset.get_training_example, invalid_index)

        # check that correct training example is retrieved
        valid_index = 0
        training_example = self.sharded_attention_dataset.get_training_example(valid_index)
        self.assertEqual(valid_index, training_example.label)

    def test_get_training_examples(self):
        """Test annotation extraction for single sequence."""

        # check that key error is raised for incorrect indices
        invalid_indices = [3]
        self.assertRaises(KeyError, self.sharded_attention_dataset.get_training_examples, invalid_indices)

        # check that correct training examples are retrieved
        valid_indices = [0, 1]
        indexed_training_examples = zip(valid_indices, self.sharded_attention_dataset.get_training_examples(valid_indices))
        for (idx, te) in indexed_training_examples:
            self.assertEqual(idx, te.label)


if __name__ == '__main__':
    unittest.main()
