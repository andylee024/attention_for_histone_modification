import argparse
import logging
import os
import sys
import tensorflow as tf

from komorebi.libs.dataset.types.tf_dataset_wrapper import tf_dataset_wrapper 
from komorebi.libs.evaluation.model_scoring import score_multitask_model, score_single_task_model
from komorebi.libs.model.attention_model import AttentionModel
from komorebi.libs.utilities.create_config import create_dataset_configuration
from komorebi.libs.utilities.io_utils import ensure_directory, write_object_to_disk

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

INFERENCE_SET = "inference_set.pkl"
TASK_METRICS = "metrics.txt"

def main(args):
    """Evaluate model."""
    _setup_device_environment(args)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        dataset = _load_dataset(args.dataset)
        model = _load_model(args, sess)
        sess.run(init_op)
        task_metrics, inference_set = score_single_task_model(args.task_index, model, dataset, sess)
    
    if args.results_directory:
        ensure_directory(args.results_directory, logger)
        _save_results(args.task_index, task_metrics, inference_set, args.results_directory)

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _setup_device_environment(args):
    """Setup os environment to run either using cpu or gpu."""
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _save_results(task_id, task_metrics, inference_set, results_directory):
    """Save results of evaluation in results directory.
    
    :param task_metrics: task metrics object computed for model and dataset
    :param inference_set: inference set object
    :param results_directory: directory for which to save results
    """
    task_id_tag = "task_{}_".format(task_id)
    inference_set_path = os.path.join(results_directory, task_id_tag + INFERENCE_SET)
    task_metrics_path = os.path.join(results_directory, task_id_tag + TASK_METRICS)

    write_object_to_disk(inference_set, inference_set_path, logger)
    _save_task_metrics(task_metrics, task_metrics_path)


def _save_task_metrics(task_metrics, path):
    """Save task metrics to a .txt file.
    
    :param task_metrics: task metrics object computed for model and dataset
    :param path: path for which to save task metrics
    """
    with open(path, 'w') as f:
        f.write(str(task_metrics))


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

# ----------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for evaluating trained tensorflow models.")

    parser.add_argument("-task", "--task-index", type=int, required=True, help="Task index for which to evaluate model on.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Configuration json for dataset.")
    parser.add_argument("-r", "--results-directory", type=str, default=None, required=False, help="Path to directory for storing results.")

    model_parser = parser.add_mutually_exclusive_group()
    model_parser.add_argument("-pb", "--trained-model-directory", type=str, help="Directory containing trained .pb model file.")
    model_parser.add_argument("-ck", "--checkpoint-model", type=str, help="Path to model checkpoint.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true", help="If set, run tensorflow with cpu.")
    mode.add_argument("--gpu", action="store_true", help="If set, run tensorflow with gpu.")

    args = parser.parse_args(sys.argv[1:])
    main(args)
