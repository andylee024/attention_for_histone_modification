#
# Attention for Histone Modification
#

from attention_for_histone_modification.libs.preprocessing.attention_types import (
        AttentionDataset, AttentionDatasetConfig, AttentionTrainingExample)

def create_single_example_dataset_with_label(label):
    """Create a dataset with a single training example with specified label.

    This test utility assumes that the global training_example index across all datasets 
    is equivalent to the supplied label.
    
    :param label: Int. 
    :return: AttentionDataset object.
    """
    return AttentionDataset(config=create_attention_config_by_indices([label]),
                            training_examples=[create_training_example_by_label(label)])

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

