# 
# Attention for Histone Modification
# 

from keras.models import Sequential, Model
import collections
import scipy.io
import numpy as np

from danq import get_trained_danq_model

training_example = collections.namedtuple(
        typename="training_example", field_names=['sequence', 'annotation_vector', 'label'])

dataset = collections.namedtuple(
        typename="dataset", field_names=['training_examples'])

def get_annotation_vector(x, model):
    """Get annotation vector.
    
    :param x:
        training sequence
    :param model:
        CNN model
    :param layer_name:
        layer name for which to extract annotaiton
    """
    #training_sequence = np.transpose(x)
    training_sequence = np.expand_dims(x, axis=0)
    return model.predict(training_sequence).flatten()

def generate_attention_training_example(x, y, model):
    """Return training example."""
    return training_example(sequence=x, label=y, annotation_vector=get_annotation_vector(x, model))

#def generate_attention_dataset(X, Y, model):
#    """Generates attention dataset.
#
#    :param X:
#        Numpy array representing X data.
#    :param Y:
#        Numpy array representing Y data.
#    :param model:
#        CNN from which to extract annotation vectors.
#    :return:
#        dataset object
#    """
#    number_examples, _, _ = X.shape
#    training_examples = [ for k in xrange(number_examples)]
#    return dataset(training_examples=training_examples)

def main():
    """Generate dataset for path."""
    
    # initialize danq model
    danq_weights = '/Users/andy/Projects/bio_startup/research/attention_for_histone_modification/experimental/danq_weights.hdf5'
    danq_model = get_trained_danq_model(danq_weights)
    annotation_vector_model = Model(inputs=danq_model.input,
                                    outputs=danq_model.get_layer("dense_1").output)

    # create training example
    dummy_training_sequence = np.zeros(shape=(1000, 4))
    dummy_label = np.zeros(919)

    training_example = generate_attention_training_example(dummy_training_sequence, dummy_label, annotation_vector_model)
    assert training_example.annotation_vector.size == 925

if __name__ == "__main__":
    main()









