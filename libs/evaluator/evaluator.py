import numpy as np
import sklearn.metrics
import tensorflow as tf
from tqdm import trange

from komorebi.libs.evaluator.inference_set import multitask_inference_set, single_task_inference_set
from komorebi.libs.evaluator.metric_types import attention_interpretation, multitask_validation_point, validation_point
from komorebi.libs.evaluator.metrics import compute_task_metrics
from komorebi.libs.trainer.trainer_utils import get_data_stream_for_epoch
from komorebi.libs.utilities.constants import TOTAL_DEEPSEA_TASKS


class Evaluator(object):
    """Class for evaluating tensorflow models on a dataset."""
    
    def __init__(self):
        self._inference_set = None

    def score_multitask_model(self, model, dataset, sess):
        """Evaluate multitask model on dataset.
        
        :param model: trained tensorflow model satisfying abstract model interface
        :param dataset: tf_dataset_wrapper object of dataset on which to evaluate model
        :param sess: tensorflow session
        """
        self._inference_set = multitask_inference_set(total_tasks=TOTAL_DEEPSEA_TASKS)
        data_stream_op = _build_evaluation_datastream(dataset, sess)

        for _ in trange(dataset.number_of_examples, desc="evaluation_progress"):
            multitask_vp = _convert_example_to_multitask_validation_point(
                    model=model, training_example=sess.run(data_stream_op), sess=sess)
            self._inference_set.add_multitask_validation_point(multitask_vp)

        return compute_task_metrics(self._inference_set)

    def score_single_task_model(self, task_id, model, dataset, sess):
        """Evaluate single task model on dataset.

        :param task_id: task_id that identifies predictive task
        :param model: trained tensorflow model satisfying abstract model interface
        :param dataset: tf_dataset_wrapper object of dataset on which to evaluate model
        :param sess: tensorflow session
        """
        self._inference_set = single_task_inference_set(task_id=task_id, task_name="unknown")
        data_stream_op = _build_evaluation_datastream(dataset, sess)

        for _ in trange(dataset.number_of_examples, desc="evaluation_progress"):
            singletask_vp = _convert_example_to_single_task_validation_point(
                    task_id=task_id, model=model, training_example=sess.run(data_stream_op), sess=sess)
            self._inference_set.add_validation_point(singletask_vp)
            
            print "sequence: {}".format(_convert_to_letters(singletask_vp.attention_interpretation.sequence))
            print "classification: {}".format(singletask_vp.classification)
            print "probability_prediction: {}".format(singletask_vp.probability_prediction)
            print "context_probabilities: {}".format(singletask_vp.attention_interpretation.context_probabilities)
            
            context_probabilities = singletask_vp.attention_interpretation.context_probabilities
            max_index = np.argmax(context_probabilities)

            start_idx = max(0, max_index - 6)
            end_idx = min(len(singletask_vp.attention_interpretation.sequence), max_index + 7)
            print "influence_motif: {}".format(_convert_to_letters(singletask_vp.attention_interpretation.sequence[start_idx:end_idx]))


        return compute_task_metrics(self._inference_set)

# DEMO
def _convert_to_letters(sequence):
    d = {0: 'a', 1: 'c', 2: 'g', 3: 't'}
    return "".join([d[s] for s in sequence])

def _build_evaluation_datastream(dataset, sess):
    """Build dataset iterator for evaluation of model.

    The present implementation builds an iterator that cycles through training examples
    one by one.

    :param dataset: tensorflow dataset wrapper
    :param sess: tensorflow session
    :return: datastream op for querying examples
    """
    dataset.build_input_pipeline_iterator(batch_size=1, buffer_size=100, parallel_calls=2)
    data_stream_op = get_data_stream_for_epoch(dataset, sess)
    return data_stream_op


def _convert_example_to_single_task_validation_point(task_id, model, training_example, sess):
    """Evaluate single training example for particular task.

    :param task_id: task_id that identifies predictive task
    :param model: trained tensorflow model satisfying abstract model interface
    :param training_example: single training example from dataset
    :param sess: tensorflow session
    :return: validation point for training example
    """
    label = np.ravel(training_example['label'])
    single_task_label = label[task_id]
    context_probabilities, probability, classification = sess.run(
            [model.inference['context_probabilities'], model.inference['prediction'], model.inference['classification']],
            feed_dict={model.inputs['sequence']: training_example['sequence'],
                       model.inputs['features']: training_example['annotation']})

    return validation_point(classification=np.ravel(classification),
                            probability_prediction=np.ravel(probability),
                            label=np.ravel(single_task_label),
                            attention_interpretation_info=attention_interpretation(
                                sequence=np.ravel(training_example['sequence']),
                                context_probabilities=np.ravel(context_probabilities)))


def _convert_example_to_multitask_validation_point(model, training_example, sess):
    """Evaluate single training example.
    
    :param model: trained tensorflow model statisfying abstract model interface
    :param training_example: single training example from dataset
    :return: score struct for training example
    """
    probabilities, classification = sess.run(
            [model.inference['prediction'], model.inference['classification']],
            feed_dict={model.inputs['sequence']: training_example['sequence'],
                       model.inputs['features']: training_example['annotation']})
    
    # convert to 1-D arrays
    probabilities = np.ravel(probabilities)
    classifications = np.ravel(classification)
    labels = np.ravel(training_example['label'])
    
    return multitask_validation_point(
            classifications=classifications, probability_predictions=probabilities, labels=labels)
