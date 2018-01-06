import numpy as np
import os
import sklearn.metrics
import tensorflow as tf

from komorebi.libs.dataset.types.dataset_config import DatasetConfiguration
from komorebi.libs.dataset.types.tf_dataset_wrapper import tf_dataset_wrapper 
from komorebi.libs.model.attention_model import AttentionModel
from komorebi.libs.trainer.trainer_utils import compute_number_of_batches, get_data_stream_for_epoch

TRAINED_MODEL_DIRECTORY = "/tmp/attention_experiment_test/trained_model"
TF_VALIDATION_DATSET = "/Users/andy/Projects/attention_histone_modification/datasets/attention_validation_tf_dataset"

def _load_dataset():
    """Load tensorflow dataset for evaluation."""
    dataset_config = DatasetConfiguration(dataset_name='validation_dataset', 
                                          examples_directory=TF_VALIDATION_DATSET)
    return tf_dataset_wrapper(dataset_config)

def _load_model():
    return AttentionModel()

def main():
    # setup evaluation objects
    dataset = _load_dataset()
    model = _load_model()

    # setup evaluation ops
    dataset.build_input_pipeline_iterator(batch_size=1, buffer_size=100, parallel_calls=2)
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)

        model.load_trained_model(TRAINED_MODEL_DIRECTORY, sess)
        data_stream_op = get_data_stream_for_epoch(dataset, sess)
        
        # run evaluation
        data = sess.run(data_stream_op)


        classification = sess.run(
                model.inference['classification'], 
                feed_dict={
                    model.inputs['sequence']: data['sequence'],
                    model.inputs['features']: data['annotation']})


        print "predictions_shape: {}".format(classification.shape)
        print "labels_shape: {}".format(data['label'].shape)

        accuracy = sklearn.metrics.accuracy_score(np.ravel(data['label']), np.ravel(classification), normalize=True)
        print "accuracy: {}".format(accuracy)
        
        accuracy_unflat = sklearn.metrics.accuracy_score(data['label'], classification, normalize=True)
        print "unflattened accuracy:{}".format(accuracy_unflat)
    
    
if __name__ == "__main__":
    main()



