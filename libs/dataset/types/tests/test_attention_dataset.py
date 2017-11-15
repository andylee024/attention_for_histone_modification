import unittest

from komorebi.libs.dataset.types.attention_dataset import (
        AttentionDataset, AttentionDatasetConfig)
from komorebi.libs.dataset.types.attention_training_example import AttentionTrainingExample
from komorebi.libs.dataset.types.tests.unittest_helpers import (
        create_attention_config_by_indices, create_training_example_by_label)

class TestAttentionDataset(unittest.TestCase):
    """Tests for extracting annotation vectors."""

    def setUp(self):
        """Initialize  attention dataset for tests."""
        indices = [0, 1, 2]
        training_examples = [create_training_example_by_label(idx) for idx in indices]
        attention_config = create_attention_config_by_indices(indices)

        self.dataset = AttentionDataset(config=attention_config, training_examples=training_examples)
        
    def test_get_training_examples(self):
        """Test annotation extraction for single sequence."""

        # method throws when supplied with incorrect indices
        invalid_indices = [4, 5]
        self.assertRaises(IndexError, self.dataset.get_training_examples, invalid_indices)

        # method returns correct indices
        query_indices = [0, 1, 2]
        indexed_training_examples = self.dataset.get_training_examples(query_indices)
        for (index, te) in zip(query_indices, indexed_training_examples):
            self.assertEqual(index, te.label)

if __name__ == '__main__':
    unittest.main()
