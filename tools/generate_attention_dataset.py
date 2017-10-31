#
# Attention for Histone Modification
#

import argparse
import logging
import json
import numpy as np
import os
import pickle
import shutil
import sys
from tqdm import tqdm

from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.batch_processing import (
        partition_and_annotate_data, create_dataset_from_attention_partition)
from attention_for_histone_modification.libs.preprocessing.attention_types import (
        AttentionDatasetConfig, AttentionDataset, AttentionTrainingExample)
from attention_for_histone_modification.libs.preprocessing.sharded_attention_dataset import AttentionDatasetInfo 

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ATTENTION_DATASET_DIRECTORY_NAME = "attention_sharded_datasets"
ATTENTION_DATASET_INFO_DIRECTORY_NAME = "attention_sharded_datasets_info"

def main(args):

    attention_config = _create_attention_config_from_json(args.config)
    dataset_directory = os.path.join(os.path.abspath(args.directory), attention_config.dataset_name)

    if args.dry_run:
        logger.info("Initiating dry_run routine...")
        logger.info("Datasets not created but will be stored at : {}".format(dataset_directory))
        exit(0)

    else:
        _handle_overwrite(dataset_directory, args.overwrite)
        _handle_directory_creation(dataset_directory)
        _handle_raw_data_copy(attention_config, dataset_directory, args.copy_data)
        _handle_dataset_generation(attention_config, dataset_directory)

# ------------------------------------------------------------------------------------------------------------
# Logic for dataset generation 
# ------------------------------------------------------------------------------------------------------------

def _handle_overwrite(dataset_directory, overwrite=False):
    """Overwrite of specified directory if overwrite flag is set.

    :param dataset_directory: Path to dataset directory.
    :param overwrite: overwrite flag option.
    """
    if overwrite and os.path.isdir(dataset_directory):
        logger.info("--overwrite flag set, Initiating overwrite routine...")
        logger.info("\t deleting {}".format(dataset_directory))
        shutil.rmtree(dataset_directory)


def _handle_directory_creation(dataset_directory):
    """Create directories needed for data generation tool.
    
    The directory structure looks like

    |-- dataset_directory
        |-- attention_datasets
        |-- attentioN_datasets_info
    """
    logger.info("Initiating directory creation routine...")
    os.mkdir(dataset_directory)
    os.mkdir(os.path.join(dataset_directory, ATTENTION_DATASET_DIRECTORY_NAME))
    os.mkdir(os.path.join(dataset_directory, ATTENTION_DATASET_INFO_DIRECTORY_NAME))


def _handle_raw_data_copy(attention_config, dataset_directory, copy_data=False):
    """Copy raw data to dataset directory.

    If the copy_data flag is set to true, then copy all raw data specified in attention_config
    into dataset directory, including the configuration file. This operation ensures reproducibility.
    """
    raw_data_directory = os.path.join(dataset_directory, "raw_data")
    if copy_data:
        logger.info("--copy-data flag set, initiating copy routine...")
        os.mkdir(raw_data_directory)
        _copy_data(attention_config.sequence_data, raw_data_directory)
        _copy_data(attention_config.label_data, raw_data_directory)
        _copy_data(attention_config.model_weights, raw_data_directory)


def _handle_dataset_generation(attention_config, dataset_directory, partition_size=1000):
    """Dataset creation procedure.

    :param attention_config: configuration object for generating attention dataset.
    :param dataset_directory: directory where datasets are saved.
    :param partition_size: specifies number of training examples datasets that are saved to disk contains.
    """
    logger.info("Initiating dataset generation routine...")

    logger.info("\t Loading annotation extractor...")
    extractor = AnnotationExtractor(model=get_trained_danq_model(attention_config.model_weights),
                                    layer_name=attention_config.model_layer)

    logger.info("\t Loading sequences...")
    sequences = np.load(attention_config.sequence_data)

    logger.info("\t Loading labels...")
    labels = np.load(attention_config.label_data)

    logger.info("\t Creating partition stream...")
    attention_partition_stream, total_partitions = partition_and_annotate_data(sequences=sequences,
                                                                               labels=labels,
                                                                               extractor=extractor,
                                                                               partition_size=partition_size)

    logger.info("\t Creating dataset stream...")
    dataset_stream = (create_dataset_from_attention_partition(attention_config, ap)
                      for ap in attention_partition_stream)

    logger.info("\t Generating datasets...")
    for dataset in tqdm(dataset_stream, total=total_partitions):
        _handle_dataset_write_logic(dataset, dataset_directory)


def _handle_dataset_write_logic(attention_dataset, dataset_directory):
    """Handle logic associated with writing datasets and other associated objects to disk.

    :param attention_dataset: instance of attention dataset
    :param dataset_directory: base directory that contains generated data.
    """
    # prepare and write attention dataset
    attention_dataset_id = "{}.pkl".format(attention_dataset.config.dataset_name)
    attention_dataset_path = os.path.join(
            dataset_directory, ATTENTION_DATASET_DIRECTORY_NAME, attention_dataset_id)
    
    with open(attention_dataset_path, 'w') as f:
        pickle.dump(attention_dataset, f)
        logger.info("\t Saved {}".format(attention_dataset_path))

    # parepare and write attention dataset info
    attention_info_id = "{}_info.pkl".format(attention_dataset.config.dataset_name)
    attention_info_path = os.path.join(
        dataset_directory, ATTENTION_DATASET_INFO_DIRECTORY_NAME, attention_info_id)
    attention_info = AttentionDatasetInfo(dataset_path=attention_dataset_path,
                                          indices=attention_dataset.config.indices)

    with open(attention_info_path, 'w') as f:
        pickle.dump(attention_info, f)
        logger.info("\t Saved {}".format(attention_info_path))


# ------------------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------------------

def _copy_data(src, dst):
    """Copy src file to dst directory and log information.
    
    :param src: file to copy
    :param dst: destination directory
    """
    assert os.path.isfile(src)
    assert os.path.isdir(dst)
    shutil.copy(src, dst)
    logger.info("\t copied {} to {}".format(os.path.basename(src), dst))


def _create_attention_config_from_json(json_path):
    """Validate and load attention dataset config from json file.

    :param config_path: path to dataset json config
    :return: dictionary containing dataset information
    """
    with open(json_path, 'r') as f:
        dataset_information = json.load(f)
        return AttentionDatasetConfig(dataset_name=dataset_information['dataset_name'],
                                      sequence_data=dataset_information['sequence_data'],
                                      label_data=dataset_information['label_data'],
                                      indices=None,
                                      model_name=dataset_information['model_name'],
                                      model_weights=dataset_information['model_weights'],
                                      model_layer=dataset_information['model_layer'])

def _get_dataset_path(directory, dataset_name):
    """Return dataset path.

    :param directory: Path to directory where datset is stored.
    :param dataset_name: name of dataset
    :return: Path to saved dataset.
    """
    return os.path.join(directory, "{}.pkl".format(dataset_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for extracting data from deepsea dataset.")
    parser.add_argument("-c", "--config", type=str, required=True, help="configuration json for dataset generation.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to output directory for datasets.")
    parser.add_argument("--dry-run", action="store_true", help="If set, do not create dataset just return path.")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite dataset directory if it exists.")
    parser.add_argument("--copy-data", action="store_true", help="If set, copy all raw data to output directory.")
    
    args = parser.parse_args(sys.argv[1:])
    main(args)
