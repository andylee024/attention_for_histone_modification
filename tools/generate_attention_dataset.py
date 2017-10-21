#
# Attention for Histone Modification
#

import argparse
import numpy as np
import os
import pickle
import sys
import time
from tqdm import tqdm

from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.ml_types import (AttentionDatasetConfig, AttentionDataset, AttentionTrainingExample)
from attention_for_histone_modification.libs.utilities.profile import time_function


def main(args):

    attention_config = AttentionDatasetConfig(args.config)
    dataset_path = _get_dataset_path(args.directory, attention_config.dataset_name)

    if args.dry_run:
        print "Dry run... not actually creating dataset."
        print "dataset path: {}".format(os.path.abspath(dataset_path))
    
    else:
        print "Starting dataset generation... \n"

        extractor = AnnotationExtractor(model=get_trained_danq_model(attention_config.model_weights),
                                        layer_name=attention_config.model_layer)

        sequences = np.load(attention_config.sequence_data)
        labels = np.load(attention_config.label_data)
        annotations = _get_annotations(sequences, extractor)

        dataset = _convert_to_attention_dataset(sequences, labels, annotations)
        print "finished generating dataset."



def _get_dataset_path(directory, dataset_name):
    """Return dataset path.
    
    :param directory: Path to directory where datset is stored.
    :param dataset_name: name of dataset
    :return: Path to saved dataset.
    """
    return os.path.join(directory, "{}.pkl".format(dataset_name))


@time_function
def _get_annotations(sequences, extractor):
    """Get annotations corresponding to sequences.

    :param sequences:
        Numpy array containing sequence data of shape (number_examples, sequence_length, vocabulary_size)
    :param extractor:
        Extractor object to get annotations from sequences.
    :return annotations:
        Numpy array containing annotations of shape (number_examples, annotation_dimension)
    """
    return extractor.extract_annotation_batch(sequences)


def _convert_to_attention_dataset(sequences, labels, annotations):
    """Extracts data from deepsea dataset.

    :param sequences:
        Numpy array containing sequence data of shape (number_examples, sequence_length, vocabulary_size)
    :param labels:
        Numpy array containing labels of shape (number_examples, label_dimension)
    :param annotations:
        Numpy array containing annotations of shape (number_examples, annotation_dimension)
    :return:
        List of training examples.
    """
    # validate sequences and labels and annotations have correct number of examples
    number_examples, _, _ = sequences.shape
    assert all((array.shape[0] == number_examples) for array in [sequences, labels, annotations])

    # construct training examples
    print "generating training examples ..."
    training_examples = [
        AttentionTrainingExample(sequence=s, label=l, annotation=a)
        for (s, l, a) in tqdm(zip(sequences, labels, annotations))]

    return AttentionDataset(training_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for extracting data from deepsea dataset.")
    parser.add_argument("-c", "--config", type=str, required=True, help="configuration json for dataset generation.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to output directory for datasets.")
    parser.add_argument("--dry-run", action="store_true", help="If set, do not create dataset just return path.")
    parser.add_argument("--gpu", action="store_true", help="If set, run using a GPU.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
