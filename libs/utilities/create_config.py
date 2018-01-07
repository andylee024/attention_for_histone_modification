"""Utilities for creating configuration files."""

import json

from komorebi.libs.dataset.types.dataset_config import DatasetConfiguration
from komorebi.libs.model.attention_configuration import AttentionConfiguration
from komorebi.libs.optimizer.optimizer_config import OptimizerConfiguration
from komorebi.libs.trainer.trainer_config import TrainerConfiguration

def create_dataset_configuration(dataset_config_json, logger=None):
    """Create dataset configuration given dataset json.
    
    :param dataset_config_json: json file specifying dataset configurations
    :param logger: if supplied, then log status
    :return: dataset configuration object
    """
    if logger:
        logger.info("\t Creating dataset configuration...")

    with open(dataset_config_json, 'r') as f:
        datastore = json.load(f)
        return DatasetConfiguration(dataset_name=datastore['dataset_name'], 
                                    examples_directory=datastore['examples_directory'])


def create_model_configuration(model_config_json, logger=None):
    """Create model configuration given model json.
    
    :param model_json: json file specifying model configuration
    :param logger: if supplied, then log status
    :return: model configuration object
    """
    if logger:
        logger.info("\t Creating model configuration...")

    with open(model_config_json, 'r') as f:
        datastore = json.load(f)
        if datastore['model_type'] == "attention":
            return AttentionConfiguration(sequence_length=datastore['sequence_length'],
                                          vocabulary_size=datastore['vocabulary_size'],
                                          prediction_classes=datastore['prediction_classes'],
                                          number_of_annotations=datastore['number_of_annotations'],
                                          annotation_size=datastore['annotation_size'],
                                          hidden_state_dimension=datastore['hidden_state_dimension'])
        else:
            raise NotImplementedError("model_type = {} not recognized!".format(datastore['model_type']))


def create_optimizer_configuration(trainer_config_json, logger=None):
    """Create optimizer configuration.

    Note, that the optimizer config settings are contained within the trainer config. 
    
    :param trainer_config_json: json file specifying trainer configurations
    :param logger: if supplied, then log status
    :return: optimizer config
    """
    if logger:
        logger.info("\t Creating optimizer configuration...")

    with open(trainer_config_json, 'r') as f:
        datastore = json.load(f)
        return OptimizerConfiguration(optimizer_type=datastore['optimizer']['type'], 
                                      learning_rate=datastore['optimizer']['learning_rate'])


def create_trainer_configuration(trainer_config_json, logger=None):
    """Create trainer configuration from json.

    :param trainer_config_json: json file specifying trainer configuration.
    :param logger: if supplied, then log status
    :return: trainer configuration object.
    """
    if logger:
        logger.info("\t Creating trainer configuration...")

    with open(trainer_config_json, 'r') as f:
        datastore = json.load(f)
        return TrainerConfiguration(epochs=datastore['epochs'],
                                    batch_size=datastore['batch_size'],
                                    buffer_size=datastore['buffer_size'],
                                    parallel_calls=datastore['parallel_calls'],
                                    checkpoint_frequency=datastore['checkpoint_frequency'])
