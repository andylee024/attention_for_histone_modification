#
# Attention for Histone Modification
#

import argparse
import logging
import json
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.batch_processing import (
        partition_and_annotate_data, create_dataset_from_attention_partition)
from attention_for_histone_modification.libs.preprocessing.attention_types import (
    AttentionDatasetConfig, AttentionDataset, AttentionTrainingExample)

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(args):

    attention_config = load_attention_config_from_json(args.config)
    dataset_path = _get_dataset_path(
        args.directory, attention_config.dataset_name)

    if args.dry_run:
        logger.info("Dry run... not actually creating dataset.")
        logger.info("dataset path: {}".format(os.path.abspath(dataset_path)))

    else:
        logger.info("Starting dataset generation...")

        logger.info("Loading annotation extractor...")
        extractor = AnnotationExtractor(model=get_trained_danq_model(attention_config.model_weights),
                                        layer_name=attention_config.model_layer)

        logger.info("Loading sequences...")
        sequences = np.load(attention_config.sequence_data)

        logger.info("Loading labels...")
        labels = np.load(attention_config.label_data)

        logger.info("Creating partition stream...")
        attention_partition_stream = partition_and_annotate_data(sequences=sequences,
                                                                 labels=labels,
                                                                 extractor=extractor,
                                                                 partition_size=1000)

        logger.info("Creating dataset stream...")
        dataset_stream = (create_dataset_from_attention_partition(attention_config, ap)
                          for ap in attention_partition_stream)

        logger.info("Generating datasets...")
        for dataset in tqdm(dataset_stream, total=8):
            logger.info("\t Saving {}".format(dataset.config.dataset_name))


def load_attention_config_from_json(json_path):
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
        #_validate_dataset_information(dataset_information)

def _get_dataset_path(directory, dataset_name):
    """Return dataset path.

    :param directory: Path to directory where datset is stored.
    :param dataset_name: name of dataset
    :return: Path to saved dataset.
    """
    return os.path.join(directory, "{}.pkl".format(dataset_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command line tool for extracting data from deepsea dataset.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="configuration json for dataset generation.")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Path to output directory for datasets.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not create dataset just return path.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="If set, run using a GPU.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
