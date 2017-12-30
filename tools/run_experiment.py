import argparse
import collections
import json
import logging
import os
import sys

from komorebi.libs.model.attention_configuration import AttentionConfiguration
from komorebi.libs.model.attention_model import AttentionModel 
from komorebi.libs.dataset.dataset_config import DatasetConfiguration
from komorebi.libs.dataset.types.tf_dataset_wrapper import tf_dataset_wrapper 
from komorebi.libs.model.parameter_initialization import ParameterInitializationPolicy
from komorebi.libs.optimizer.optimizer_config import OptimizerConfiguration
from komorebi.libs.optimizer.optimizer_factory import create_tf_optimizer 
from komorebi.libs.trainer.attention_trainer import AttentionTrainer
from komorebi.libs.trainer.trainer_config import TrainerConfiguration
from komorebi.libs.utilities.io_utils import copy_data, ensure_directory, load_pickle_object, remove_directory

# Tensorflow specific directories
CHECKPOINT_DIRECTORY_NAME = "model_checkpoints"
CONFIG_DIRECTORY_NAME = "config"
SUMMARY_DIRECTORY_NAME = "training_summaries"

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Struct storing configuration information needed for specifying ml experiment.
ExperimentConfiguration = collections.namedtuple(typename='ExperimentConfiguration', 
                                                 field_names=['experiment_directory',
                                                              'checkpoint_directory',
                                                              'config_directory',
                                                              'summary_directory',
                                                              'dataset_config', 
                                                              'model_config', 
                                                              'trainer_config', 
                                                              'optimizer_config', 
                                                              'parameter_initialization'])

def main(args):
    """Launch machine learning tensorflow experiment."""

    logger.info("Launching tensorflow experiment...")
    experiment_config = _parse_experiment_config_json(args.config)
    _handle_overwrite(experiment_config.experiment_directory, args.overwrite)
    _handle_directory_creation(experiment_config)
    _handle_config_copy(args.config, experiment_config.config_directory)

    logger.info("Loading machine learning objects ...")
    dataset = _load_dataset(experiment_config.dataset_config)
    model = _load_model(experiment_config.model_config, experiment_config.parameter_initialization)
    optimizer = _load_optimizer(experiment_config.optimizer_config)
    trainer = _load_trainer(experiment_config)

    # train model
    trainer.train_model(tf_dataset_wrapper=dataset, model=model, optimizer=optimizer)

# ----------------------------------------------------------------
# Helpers for setting up experiment
# ----------------------------------------------------------------
def _parse_experiment_config_json(experiment_config_json):
    """Parse experiment json and convert to experiment config.

    Parse relevant fields in experiment configuration file. 

    :param config_json: path to experiment config json.
    :return: 
        ExperimentConfiguration named-tuple with the following attributes.
            experiment_name      : name of experiment
            experiment_directory : base directory to create experiment directory 
            dataset_config              : dataset configuration object
            model_config                : model configuration object
            trainer_config              : trainer configuration object
            optimizer_config            : optimizer configuration object
    """

    logger.info("Parsing experiment configuration parameters...")

    with open(experiment_config_json, 'r') as f:
        experiment_info = json.load(f)

        experiment_directory = os.path.join(experiment_info['experiments_directory'], 
                                            experiment_info['experiment_name'])
        checkpoint_directory = os.path.join(experiment_directory, CHECKPOINT_DIRECTORY_NAME)
        config_directory = os.path.join(experiment_directory, CONFIG_DIRECTORY_NAME)
        summary_directory = os.path.join(experiment_directory, SUMMARY_DIRECTORY_NAME)
        
        return ExperimentConfiguration(experiment_directory=experiment_directory,
                                       checkpoint_directory=checkpoint_directory,
                                       config_directory=config_directory,
                                       summary_directory=summary_directory,
                                       parameter_initialization=ParameterInitializationPolicy(),
                                       dataset_config=_create_dataset_configuration(experiment_info['dataset_config']),
                                       model_config=_create_model_configuration(experiment_info['model_config']),
                                       trainer_config=_create_trainer_configuration(experiment_info['trainer_config']),
                                       optimizer_config=_create_optimizer_configuration(experiment_info['trainer_config']))


def _handle_overwrite(experiment_directory, overwrite_flag=False):
    """Overwrite of specified directory if overwrite flag is set.

    :param experiment_directory: Path to experiment directory.
    :param overwrite: overwrite flag option.
    """
    if overwrite_flag:
        logger.info("--overwrite flag set, Initiating overwrite routine...")
        remove_directory(experiment_directory, logger=logger)


def _handle_directory_creation(experiment_config):
    """Create directories for machine learning experiment.

    The directory structure created for a single experiment is given as follows.

    |-- experiment_directory (holds experiment specific data)
        |-- config (holds configs)
        |-- checkpoints (holds model checkpoints)
        |-- trained_model (holds trained tf model)
        |-- training_summaries (holds tf summaries)
    
    :param experiment_config: config object with directory information
    """
    logger.info("Initiating directory creation routine ...")

    if os.path.isdir(experiment_config.experiment_directory):
        raise IOError("Experiment directory: {} already exists!".format(experiment_config.experiment_directory))
    else:
        ensure_directory(experiment_config.experiment_directory)
        ensure_directory(experiment_config.checkpoint_directory)
        ensure_directory(experiment_config.config_directory)
        ensure_directory(experiment_config.summary_directory)


def _handle_config_copy(experiment_config_json, experiment_config_directory):
    """Copy configuration files to experiment directory.

    :param experiment_config_json: path to experiment config json.
    :param experiment_config_directory: directory for storing experiment configs
    """
    logger.info("Copying experiment configs ...")

    with open(experiment_config_json, 'r') as f:
        experiment_info = json.load(f)
        copy_data(source=experiment_info['dataset_config'], destination=experiment_config_directory)
        copy_data(source=experiment_info['model_config'], destination=experiment_config_directory)
        copy_data(source=experiment_info['trainer_config'], destination=experiment_config_directory)


# ----------------------------------------------------------------
# Helpers for creating configurations
# ----------------------------------------------------------------
def _create_dataset_configuration(dataset_config_json):
    """Create dataset configuration given dataset json.
    
    :param dataset_config_json: json file specifying dataset configurations
    :return: dataset configuration object
    """
    logger.info("\t Creating dataset configuration...")

    with open(dataset_config_json, 'r') as f:
        datastore = json.load(f)
        return DatasetConfiguration(dataset_name=datastore['dataset_name'], 
                                    dataset_path=datastore['dataset_path'])


def _create_model_configuration(model_config_json):
    """Create model configuration given model json.
    
    :param model_json: json file specifying model configuration
    :return: model configuration object
    """
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


def _create_optimizer_configuration(trainer_config_json):
    """Create optimizer configuration.

    Note, that the optimizer config settings are contained within the trainer config. 
    
    :param trainer_config_json: json file specifying trainer configurations
    :return: optimizer config
    """
    logger.info("\t Creating optimizer configuration...")

    with open(trainer_config_json, 'r') as f:
        datastore = json.load(f)
        return OptimizerConfiguration(optimizer_type=datastore['optimizer']['type'], 
                                      learning_rate=datastore['optimizer']['learning_rate'])


def _create_trainer_configuration(trainer_config_json):
    """Create trainer configuration from json.

    :param trainer_config_json: json file specifying trainer configuration.
    :param experiment_directory: directory storing experiment results
    :return: trainer configuration object.
    """
    logger.info("\t Creating trainer configuration...")

    with open(trainer_config_json, 'r') as f:
        datastore = json.load(f)
        return TrainerConfiguration(epochs=datastore['epochs'],
                                    batch_size=datastore['batch_size'],
                                    buffer_size=datastore['buffer_size'],
                                    parallel_calls=datastore['parallel_calls'],
                                    checkpoint_frequency=datastore['checkpoint_frequency'])

# ----------------------------------------------------------------
# Helpers for loading objects 
# ----------------------------------------------------------------

def _load_dataset(dataset_config):
    """Load dataset from config.
    
    :param dataset_config: object of type DatasetConfiguration
    :return: dataset satisfying AbstractDataset interface
    """
    logger.info("\t Loading dataset ...")
    return tf_dataset_wrapper()


def _load_model(model_config, parameter_policy):
    """Load model from config.

    :param model_config: AttentionConfiguration object
    :param parameter_policy: initialization policy for weights and biases for tensorflow models
    :return: model satisfying AbstractModel interface
    """
    logger.info("\t Loading model ...")
    return AttentionModel(attention_config=model_config, parameter_policy=parameter_policy)


def _load_optimizer(optimizer_config):
    """Load optimizer from config.

    :param optimizer_config: OptimizerConfiguration object
    :return: tensorflow optimizer object
    """
    logger.info("\t Loading optimizer ...")
    return create_tf_optimizer(optimizer_config)


def _load_trainer(experiment_config):
    """Load trainer from config.
    
    The entire experiment_config is needed to expose relevant directory information.

    :param experiment_config: experiment configuration information
    :return: trainer satisfying AbstractTensorflowTrainer interface
    """
    logger.info("\t Loading trainer ...")
    return AttentionTrainer(config=experiment_config.trainer_config,
                            experiment_directory=experiment_config.experiment_directory,
                            checkpoint_directory=experiment_config.checkpoint_directory,
                            summary_directory=experiment_config.summary_directory)
    

# ----------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for running machine learning experiments.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Configuration json for experiment.")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite experiment directory if it exists.")
    args = parser.parse_args(sys.argv[1:])
    main(args)

