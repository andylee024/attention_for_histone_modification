#
# Attention for Histone Modification
# 

import numpy as np

from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.ml_types import AttentionTrainingExample, AttentionDataset

def extract_sequences_from_deepsea_dataset(deepsea_dataset):
    """Return list of sequences from deepsea dataset."""
    return (np.expand_dims(np.transpose(s), axis=0) for s in deepsea_dataset['testxdata'][:10])

def extract_labels_from_deepsea_dataset(deepsea_dataset):
    """Return list of labels from deepsea dataset."""
    return (label for label in deepsea_dataset['testdata'][:10])

def convert_to_attention_dataset(deepsea_dataset, extractor):
    """Extracts data from deepsea dataset.

    :param data_dictionary:
        Python dictionary containing deepsea data with the following attributes.
            'testxdata' : Numpy array (batch_size, vocabulary_dimension, sequence_length)
            'testdata'  : Numpy array (batch_size, label_classes)
    :return:
        List of training examples.
    """
    sequences = list(extract_sequences_from_deepsea_dataset(deepsea_dataset))
    labels = list(extract_labels_from_deepsea_dataset(deepsea_dataset))
    annotations = [extractor.extract_annotation(s) for s in sequences]
    
    training_examples = (AttentionTrainingExample(sequence=s, label=l, annotation_vector=a) for (s, l, a) in zip(sequences, labels, annotations))
    return AttentionDataset(list(training_examples))


def main():

    import os
    import scipy.io

    # initialize dataset
    deepsea_data_path = os.path.join("/Users/andy/Projects/bio_startup/research/deepsea_data", "test.mat")
    deepsea_data = scipy.io.loadmat(deepsea_data_path)
    deepsea_data['testdata'] = deepsea_data['testdata'][:10]
    deepsea_data['testxdata'] = deepsea_data['testxdata'][:10]

    print deepsea_data['testdata'].shape
    print deepsea_data['testxdata'].shape
    
    # initialize extractor
    danq_weights = '/Users/andy/Projects/bio_startup/research/attention_for_histone_modification/experimental/danq_weights.hdf5'
    danq_model = get_trained_danq_model(danq_weights)
    extractor = AnnotationExtractor(model=danq_model, layer_name="dense_1")

    # test convert tool
    dataset = convert_to_attention_dataset(deepsea_data, extractor)
    assert isinstance(dataset, AttentionDataset)
    assert len(dataset.training_examples) == 10.0
    print "dataset generation passing ..."

if __name__ == "__main__":
    main()
    

    

