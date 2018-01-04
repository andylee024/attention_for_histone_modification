import numpy as np
import os
import tensorflow as tf

from komorebi.libs.dataset.types.dataset_config import DatasetConfiguration
from komorebi.libs.dataset.types.tf_dataset_wrapper import tf_dataset_wrapper 
from komorebi.libs.evaluator.evaluator_utils import get_inference_ops
from komorebi.libs.trainer.trainer_utils import compute_number_of_batches, get_data_stream_for_epoch
from komorebi.libs.utilities.io_utils import load_pickle_object

TRAINED_MODEL_CONFIG = "/tmp/attention_experiment_test/trained_model_config.pkl"
TF_VALIDATION_DATSET = "/Users/andy/Projects/attention_histone_modification/datasets/attention_validation_tf_dataset"

def _load_dataset():
    """Load tensorflow dataset for evaluation."""
    dataset_config = DatasetConfiguration(dataset_name='validation_dataset', 
                                          examples_directory=TF_VALIDATION_DATSET)
    return tf_dataset_wrapper(dataset_config)

def main():

    # setup evaluation objects
    dataset = _load_dataset()
    trained_model_config = load_pickle_object(TRAINED_MODEL_CONFIG)

    # setup evaluation ops
    dataset.build_input_pipeline_iterator(batch_size=1, buffer_size=100, parallel_calls=2)
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        # get ops
        inference_ops = get_inference_ops(trained_model_config, sess)
        data_stream_op = get_data_stream_for_epoch(dataset, sess)
        
        # run evaluation
        data = sess.run(data_stream_op)


        logits = sess.run(inference_ops.prediction, feed_dict={inference_ops.sequence_placeholder: data['sequence'], 
                                                               inference_ops.annotation_placeholder: data['annotation']})
        print logits.shape
    
    
if __name__ == "__main__":
    main()



