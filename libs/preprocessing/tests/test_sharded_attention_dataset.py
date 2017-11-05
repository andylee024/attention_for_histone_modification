#
# Unit tests for extractor
#
import tempfile
import unittest
import os

from attention_for_histone_modification.libs.preprocessing.sharded_attention_dataset import (
        AttentionDatasetInfo, ShardedAttentionDataset)
from attention_for_histone_modification.libs.preprocessing.utilities import (
        remove_directory, write_object_to_disk)
from attention_for_histone_modification.libs.preprocessing.tests.utilities_for_tests import (
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
        dataset_1_path = os.path.join(self.tmpdir, "dataset_1")
        dataset_2_path = os.path.join(self.tmpdir, "dataset_2")
        write_object_to_disk(obj=create_single_example_dataset_with_label(1), path=dataset_1_path)
        write_object_to_disk(obj=create_single_example_dataset_with_label(2), path=dataset_2_path)

        # create sharded attention dataset info
        index_to_dataset = {1: dataset_1_path, 2: dataset_2_path}
        self.sharded_attention_dataset = ShardedAttentionDataset(index_to_dataset)

    def tearDown(self):
        """Clean up for unit test."""
        remove_directory(self.tmpdir)

    def test_get_training_examples(self):
        """Test annotation extraction for single sequence."""

        # check that key error is raised for incorrect indices
        invalid_indices = [3]
        self.assertRaises(KeyError, self.sharded_attention_dataset.get_training_examples, invalid_indices)

        # check that correct training examples are retrieved
        valid_indices = [1, 2]
        indexed_training_examples = self.sharded_attention_dataset.get_training_examples(valid_indices) 
        for (idx, te) in indexed_training_examples:
            self.assertEqual(idx, te.label)


if __name__ == '__main__':
    unittest.main()
