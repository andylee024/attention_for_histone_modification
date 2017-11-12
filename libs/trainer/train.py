
import collections
import numpy as np
import pickle
import tensorflow as tf

from attention_for_histone_modification.libs.model.attention_configuration import (
        AttentionConfiguration, LearningConfiguration)
from attention_for_histone_modification.libs.model.attention_model import AttentionModel
from attention_for_histone_modification.libs.preprocessing.utilities import load_pickle_object, partition_indices

def main():
    dataset = _get_dataset()

    shuffled_indices = _get_shuffled_indices(dataset.total_examples)
    index_batches, _ = partition_indices(shuffled_indices, 10)
    batch_size = len(index_batches[0])

    model = _get_model(batch_size)
    trainer = attention_trainer()

    print "training model"
    trainer.train_model(model, dataset)
    return





           


def _get_dataset():
    """Get dataset used for training model.
    
    :param dataset_path: path to dataset pkl file.
    :return: AttentionDataset type object.
    """
    sharded_path = "/Users/andy/Projects/biology/research/attention_for_histone_modification/data/attention_validation_dataset/sharded_attention_dataset.pkl"
    return load_pickle_object(sharded_path)

def _get_model(batch_size):
    """Get model used for training.
    
    :return: AttentionModel
    """
    attention_config = AttentionConfiguration(batch_size=batch_size,
                                              sequence_length=1000,
                                              vocabulary_size=4,
                                              prediction_classes=919,
                                              number_of_annotations=1,
                                              annotation_size=925,
                                              hidden_state_dimension=112)
    learning_config = LearningConfiguration()
    return AttentionModel(attention_config=attention_config, learning_config=learning_config)
# ----------------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------------



def _get_shuffled_indices(number_examples):
    """Return shuffled indices for number of examples."""
    indices = np.arange(number_examples)
    np.random.shuffle(indices)
    return indices


if __name__ == "__main__":
    main()
