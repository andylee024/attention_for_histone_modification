#
# Attention for Histone Modification
# 

import argparse
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.ml_types import AttentionTrainingExample, AttentionDataset

def main(args):
    extractor = AnnotationExtractor(model=get_trained_danq_model(args.weights),
                                    layer_name=args.layer)
    sequences = np.load(args.sequences)
    labels = np.load(arg.labels)
    annotations = _get_annotations(sequeces, extractor)

    dataset = _convert_to_attention_dataset(sequences=sequences,
                                            labels=labels,
                                            annotations=annotations)
    # write dataset
    with open(os.path.join(args.directory, "{}.pkl".format(args.name)), 'w') as f:
        pickle.dump(dataset, f)


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
    assert all((len(data) for data in [sequences, labels, annotations))

    # construct training examples
    print "generating training examples..."
    training_examples = [
            AttentionTrainingExample(sequence=s, label=l, annotations=a) 
            for (s, l, a) in tqdm(zip(sequences, labels, annotations))]

    return AttentionDataset(training_examples)

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
