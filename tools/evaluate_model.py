import argparse
import sys
import tensorflow as tf

from komorebi.libs.dataset.types.tf_dataset_wrapper import tf_dataset_wrapper 
from komorebi.libs.evaluator.evaluator import Evaluator
from komorebi.libs.model.attention_model import AttentionModel
from komorebi.libs.utilities.create_config import create_dataset_configuration

def main(args):
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        dataset = _load_dataset(args.dataset)
        model = _load_model(args.model, sess)
        evaluator = _load_evaluator() 

        sess.run(init_op)
        task_metrics = evaluator.score_model(model, dataset, sess)
        print task_metrics

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _load_model(trained_model_directory, sess):
    """Return trained model associated with trained model directory.
    
    :param trained_model_directory: directory containing trained model .pb file
    :param sess: tensorflow session
    :return: model object satisfying abstract model interface
    """
    model = AttentionModel()
    model.load_trained_model(trained_model_directory, sess)
    return model


def _load_dataset(dataset_json):
    """Return tensorflow dataset associated with supplied dataset json configuration.
    
    :param dataset_json: json config file associated dataset
    :return: tensorflow dataset
    """
    dataset_config = create_dataset_configuration(dataset_json)
    return tf_dataset_wrapper(dataset_config)


def _load_evaluator():
    """Return evaluator object."""
    return Evaluator()
    
# ----------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for evaluating trained tensorflow models.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Configuration json for dataset.")
    parser.add_argument("-m", "--model", type=str, help="Directory containing trained .pb model file.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
