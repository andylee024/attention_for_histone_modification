import argparse
import os
import sys
import tensorflow as tf

from komorebi.libs.dataset.types.tf_dataset_wrapper import tf_dataset_wrapper 
from komorebi.libs.evaluator.evaluator import Evaluator
from komorebi.libs.model.attention_model import AttentionModel
from komorebi.libs.utilities.create_config import create_dataset_configuration

def main(args):
    _setup_device_environment(args)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        dataset = _load_dataset(args.dataset)
        model = _load_model(args, sess)
        evaluator = _load_evaluator() 

        sess.run(init_op)
        task_metrics = evaluator.score_single_task_model(args.task_index, model, dataset, sess)

        print task_metrics

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _setup_device_environment(args):
    """Setup os environment to run either using cpu or gpu."""
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _load_model(args, sess):
    """Return trained model associated with trained model directory.
    
    :param args: Namsspace object with the following attributes
        args.trained_model_directory: directory containing trained model .pb file
        args.checkpoint_model: directory containing checkpoint file
    :param sess: tensorflow session
    :return: model object satisfying abstract model interface
    """
    model = AttentionModel()

    if args.trained_model_directory:
        model.load_trained_model(args.trained_model_directory, sess)
    elif args.checkpoint_model:
        model.load_checkpoint_model(args.checkpoint_model, sess)
    else:
        raise ValueError("No model loading scheme specified (checkpoint or trained).")

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

    parser.add_argument("-task", "--task-index", type=int, required=True, help="Task index for which to evaluate model on.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Configuration json for dataset.")

    model_parser = parser.add_mutually_exclusive_group()
    model_parser.add_argument("-pb", "--trained-model-directory", type=str, help="Directory containing trained .pb model file.")
    model_parser.add_argument("-ck", "--checkpoint-model", type=str, help="Path to model checkpoint.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true", help="If set, run tensorflow with cpu.")
    mode.add_argument("--gpu", action="store_true", help="If set, run tensorflow with gpu.")

    args = parser.parse_args(sys.argv[1:])
    main(args)
