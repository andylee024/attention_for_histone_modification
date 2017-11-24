import argparse
import collections
import json
import os
import sys

from komorebi.libs.model.attention_configuration import AttentionConfiguration
from komorebi.libs.model.attention_model import AttentionModel 
from komorebi.libs.dataset.dataset_config import DatasetConfiguration
from komorebi.libs.model.parameter_initialization import ParameterInitializationPolicy
from komorebi.libs.optimizer.optimizer_config import OptimizerConfiguration
from komorebi.libs.optimizer.optimizer_factory import create_tf_optimizer 
from komorebi.libs.trainer.attention_trainer import AttentionTrainer
from komorebi.libs.trainer.trainer_config import TrainerConfiguration
from komorebi.libs.utilities.io_utils import copy_data, ensure_directory, load_pickle_object

#logging.basicConfig(format='%(asctime)s %(message)s')
#logger = logging.getLogger()
#logger.setLevel(logging.INFO)

# Struct storing configuration information needed for specifying ml experiment.
ExperimentConfiguration = collections.namedtuple(typename='ExperimentConfiguration', 
                                                 field_names=['experiment_directory',
                                                              'dataset_config', 
                                                              'model_config', 
                                                              'trainer_config', 
                                                              'optimizer_config', 
                                                              'parameter_initialization'])

def main(args):
    """Launch machine learning tensorflow experiment."""

    experiment_config = _parse_experiment_config_json(args.config)
    
    _handle_directory_creation(experiment_config.experiment_directory)
    _handle_config_copy(args.config, experiment_config.experiment_directory)

    dataset = _load_dataset(experiment_config.dataset_config)
    model = _load_model(experiment_config.model_config, experiment_config.parameter_initialization)
    optimizer = _load_optimizer(experiment_config.optimizer_config)
    trainer = _load_trainer(experiment_config.trainer_config)

    # train model
    trainer.train_model(dataset=dataset, model=model, optimizer=optimizer)


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
    with open(experiment_config_json, 'r') as f:
        experiment_info = json.load(f)
        experiment_directory = os.path.join(experiment_info['experiments_directory'], 
                                            experiment_info['experiment_name'])
        
        return ExperimentConfiguration(experiment_directory=experiment_directory,
                                       parameter_initialization=ParameterInitializationPolicy(),
                                       dataset_config=_create_dataset_configuration(experiment_info['dataset_config']),
                                       model_config=_create_model_configuration(experiment_info['model_config']),
                                       trainer_config=_create_trainer_configuration(experiment_info['trainer_config'], experiment_directory),
                                       optimizer_config=_create_optimizer_configuration(experiment_info['trainer_config']))


def _handle_directory_creation(experiment_directory):
    """Create directories for machine learning experiment."""
    if os.path.isdir(experiment_directory):
        raise IOError("Experiment directory: {} already exists!".format(experiment_directory))
    else:
        os.mkdir(experiment_directory)


def _handle_config_copy(experiment_config_json, experiment_directory):
    """Copy configuration files to experiment directory.

    :param experiment_config_json: path to experiment config json.
    :param experiment_directory: directory for storing experiment data
    """
    with open(experiment_config_json, 'r') as f:
        experiment_info = json.load(f)
        experiment_config_directory = os.path.join(experiment_directory, "config")

        ensure_directory(experiment_config_directory)
        copy_data(source=experiment_info['dataset_config'], destination=experiment_config_directory)
        copy_data(source=experiment_info['model_config'], destination=experiment_config_directory)
        copy_data(source=experiment_info['trainer_config'], destination=experiment_config_directory)


def _create_dataset_configuration(dataset_config_json):
    """Create dataset configuration given dataset json."""
    with open(dataset_config_json, 'r') as f:
        datastore = json.load(f)
        return DatasetConfiguration(dataset_name=datastore['dataset_name'], 
                                    dataset_path=datastore['dataset_path'])


def _create_model_configuration(model_config_json):
    """Create model configuration given model json.
    
    :param model_json: json file specifying model configuration.
    :return: model configuration object.
    """
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
    """Return configuration object for optimizer."""
    with open(trainer_config_json, 'r') as f:
        datastore = json.load(f)
        return OptimizerConfiguration(optimizer_type=datastore['optimizer']['type'], 
                                      learning_rate=datastore['optimizer']['learning_rate'])


def _create_trainer_configuration(trainer_config_json, experiment_directory):
    """Create trainer configuration from json.

    :param trainer_config_json: json file specifying trainer configuration.
    :param experiment_directory: directory storing experiment results
    :return: trainer configuration object.
    """
    with open(trainer_config_json, 'r') as f:
        datastore = json.load(f)
        return TrainerConfiguration(epochs=datastore['epochs'],
                                    batch_size=datastore['batch_size'],
                                    experiment_directory=experiment_directory,
                                    checkpoint_frequency=datastore['checkpoint_frequency'])


def _load_dataset(dataset_config):
    """Load dataset from config.
    
    :param dataset_config: object of type DatasetConfiguration
    :return: dataset satisfying AbstractDataset interface
    """
    return load_pickle_object(dataset_config.dataset_path)


def _load_model(model_config, parameter_policy):
    """Load model from config.

    :param model_config: AttentionConfiguration object
    :param parameter_policy: initialization policy for weights and biases for tensorflow models
    :return: model satisfying AbstractModel interface
    """
    return AttentionModel(attention_config=model_config, parameter_policy=parameter_policy)


def _load_optimizer(optimizer_config):
    """Load optimizer from config.

    :param optimizer_config: OptimizerConfiguration object
    :return: tensorflow optimizer object
    """
    return create_tf_optimizer(optimizer_config)


def _load_trainer(trainer_config):
    """Load trainer from config.

    :param trainer_config: TrainerConfiguration object
    :return: trainer satisfying AbstractTensorflowTrainer interface
    """
    return AttentionTrainer(trainer_config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for running machine learning experiments.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Configuration json for experiment.")
    args = parser.parse_args(sys.argv[1:])
    main(args)

