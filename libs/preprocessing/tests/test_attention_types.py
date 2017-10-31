#
# Unit tests for extractor
#

import numpy as np
import unittest

from attention_for_histone_modification.libs.preprocessing.attention_types import (
        AttentionDataset, AttentionDatasetConfig, AttentionTrainingExample)

class TestAttentionDataset(unittest.TestCase):
    """Tests for extracting annotation vectors."""

    @classmethod
    def setUpClass(cls):
        """Initialize  attention dataset for tests."""
        indices = range(3)
        cls.training_examples = [create_training_example_by_label(idx) for idx in indices]
        cls.attention_config = create_attention_config_by_indices(indices)

    def test_get_training_examples(self):
        """Test annotation extraction for single sequence."""
        dataset = AttentionDataset(self.attention_config, self.training_examples) 

        # method throws when supplied with incorrect indices
        invalid_indices = [4, 5]
        self.assertRaises(IndexError, dataset.get_training_examples, invalid_indices)

        # method returns correct indices
        valid_indices = [0, 1]
        training_examples = dataset.get_training_examples(valid_indices)

        expected_labels = valid_indices
        actual_labels = [te.label for te in training_examples]
        self.assertListEqual(actual_labels, expected_labels)


def create_training_example_by_label(label):
    """Create training example with only label field set."""
    return AttentionTrainingExample(sequence=None, label=label, annotation=None)

def create_attention_config_by_indices(indices):
    """Create attention configuration with only indices field set."""
    return AttentionDatasetConfig(dataset_name=None,
                                  sequence_data=None, 
                                  label_data=None,
                                  indices=indices,
                                  model_name=None,
                                  model_weights=None,
                                  model_layer=None)

if __name__ == '__main__':
    unittest.main()
