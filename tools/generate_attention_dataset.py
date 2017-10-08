#
# Attention for Histone Modification
# 

import argparse
import numpy as np
import os
import pickle
import sys

from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.ml_types import AttentionTrainingExample, AttentionDataset


def convert_to_attention_dataset(sequences, labels, extractor):
    """Extracts data from deepsea dataset.

    :param sequences:
        Numpy array containing sequence data of shape (number_examples, sequence_length, vocabulary_size)
    :param labels:
        Numpy array containing labels of shape (number_examples, label_dimension)
    :param extractor:
        AnnotationExtractor object for extracting annotation vectors from sequences.
    :return:
        List of training examples.
    """
    # validate sequences and labels are paired correctly
    assert sequences.shape[0] == labels.shape[0]

    # construct training examples
    training_examples = [
        AttentionTrainingExample(sequence=s, label=l,
                                 annotation_vector=extractor.extract_annotation(np.expand_dims(s, axis=0)))
        for (s, l) in zip(sequences, labels)]
    return AttentionDataset(training_examples)


def main(args):
    extractor = AnnotationExtractor(model=get_trained_danq_model(args.weights),
                                    layer_name=args.layer)
    dataset = convert_to_attention_dataset(sequences=np.load(args.sequences),
                                           labels=np.load(args.labels),
                                           extractor=extractor)
    pickle.dump(dataset, os.path.join(args.directory, "{}.pkl".format(args.name)))

    print "total_examples: {}".format(len(dataset.training_examples))
    print "sequence.shape: {}".format(dataset.training_examples[0].sequence.shape)
    print "label.shape: {}".format(dataset.training_examples[0].label.shape)
    print "annotation_vector.shape: {}".format(dataset.training_examples[0].annotation_vector.shape)
    print "dataset generation passing ..."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for extracting data from deepsea dataset.")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="name of dataset")
    parser.add_argument("-w", "--weights", type=str, required=True,
                        help="path to DANQ keras weights.")
    parser.add_argument("-l", "--layer", type=str, required=True,
                        help="name of layer from DANQ for which to extract annotation vectors.")
    parser.add_argument("-x", "--sequences", type=str,
                        help="Path to .npy file containing sequences (X-data).")
    parser.add_argument("-y", "--labels", type=str,
                        help="Path to .npy file containing labels (Y-data).")
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Path to output directory for saving datasets.")
    args = parser.parse_args(sys.argv[1:])
    main(args)

