# 
# Attention for Histone Modification
# 

import datetime
import json
import os


class AttentionTrainingExample(object):
    """Training example for attention models."""

    def __init__(self, sequence, label, annotation):
        """Initialize training example.

        :param sequence:
            Numpy array representing genetic sequence.
        :param label:
            Numpy array representing multi-class prediction label.
        :param annotation:
            Numpy array representing annotation vector corresponding to sequence.
        """
        self.sequence = sequence
        self.label = label
        self.annotation = annotation


class AttentionDataset(object):
    """Dataset for attention models."""

    def __init__(self, config, training_examples):
        """Initialize attention dataset.

        :param config: attention dataset config containing information about dataset.
        :param training_examples: List of generated training examples.
        """
        assert isinstance(config, AttentionDatasetConfig) 
        _validate_training_examples(training_examples)

        self.config = config
        self.training_examples = training_examples


class AttentionDatasetConfig(object):
    """Config object of attention dataset."""

    def __init__(self, config_path):
        """Initialize dataset config by populating fields.

        Attributes:
            dataset_name  : name of attention dataset
            sequence_data : path to .npy file containing sequences
            label_data    : path to .npy file containing labels
            model_name    : name of model (only DANQ right now)
            model_weights : path to weights of model
            model_layer   : name of model layer to extract annotations
            timestamp     : current timestamp

        :param config_path: path to dataset json config
        """
        dataset_information = _load_attention_json_config(config_path)

        self.dataset_name = dataset_information['dataset_name']
        self.sequence_data = dataset_information['sequence_data']
        self.label_data = dataset_information['label_data']
        self.model_name = dataset_information['model_name']
        self.model_weights = dataset_information['model_weights']
        self.model_layer = dataset_information['model_layer']
        self.timestamp = str(datetime.datetime.now())


def _validate_training_examples(training_examples):
    """Validates all training examples are correct type."""
    assert all((isinstance(te, AttentionTrainingExample) for te in training_examples))


def _validate_dataset_information(dataset_information):
    """Validate paths specified in dataset config exist."""
    assert os.path.exists(dataset_information['model_weights'])
    assert os.path.exists(dataset_information['sequence_data'])
    assert os.path.exists(dataset_information['label_data'])


def _load_attention_json_config(config_path):
    """Validate and load attention dataset config from json file.
    
    :param config_path: path to dataset json config
    :return: dictionary containing dataset information
    """
    with open(config_path, 'r') as f:
        dataset_information = json.load(f)
        _validate_dataset_information(dataset_information)
        return dataset_information

