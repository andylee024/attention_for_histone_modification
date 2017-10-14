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
from attention_for_histone_modification.libs.preprocessing.ml_types import AttentionTrainingExample, AttentionDataset
from attention_for_histone_modification.libs.utilities.profile import time_function

def main(args):

    print "Starting dataset generation... \n"

    # step 1 - extract annotations
    extractor = AnnotationExtractor(model=get_trained_danq_model(args.weights),
                                    layer_name=args.layer)
    sequences = np.load(args.sequences)
    labels = np.load(args.labels)
    annotations = _get_annotations(sequences, extractor)
   
    # step 2 - create dataset
    dataset = _convert_to_attention_dataset(sequences=sequences,
                                            labels=labels,
                                            annotations=annotations)
    # write dataset
    dataset_path = os.path.join(args.directory, "{}.pkl".format(args.name))
    with open(dataset_path, 'w') as f:
        pickle.dump(dataset, f)
        print "saved dataset {}".format(dataset_path)


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
    print "extracting annotations for {} sequences...".format(sequences.shape[0]) 
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
    assert all(array.shape for array in (sequences, labels, annotations))

    # construct training examples
    print "generating training examples progress..."
    training_examples = [
            AttentionTrainingExample(sequence=s, label=l, annotation=a) 
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
