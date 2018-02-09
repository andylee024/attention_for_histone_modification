import numpy as np
import tensorflow as tf
from tqdm import trange

from komorebi.libs.evaluation.inference_set import multitask_inference_set, single_task_inference_set
from komorebi.libs.evaluation.metric_types import attention_result, multitask_validation_point, validation_point
from komorebi.libs.trainer.trainer_utils import get_data_stream_for_epoch
from komorebi.libs.utilities.constants import TOTAL_DEEPSEA_TASKS


def generate_single_task_inference_set(task_id, model, dataset, sess):
    """Generate a single task inference set for evaluation and interpretation.
    
    :param task_id: task_id that identifies predictive task
    :param model: trained tensorflow model satisfying abstract model interface
    :param dataset: tensorflow dataset wrapper
    :param sess: tensorflow session
    :return: single task inference set object
    """
    inference_set = single_task_inference_set(task_id=task_id, task_name="unknown")
    data_stream_op = _build_evaluation_datastream(dataset, sess)

    for _ in trange(dataset.number_of_examples, desc="single task evaluation progress"):
        singletask_vp = _convert_example_to_single_task_validation_point(
                task_id=task_id, model=model, training_example=sess.run(data_stream_op), sess=sess)
        inference_set.add_validation_point(singletask_vp)

    return inference_set


def generate_multitask_inference_set(model, dataset, sess):
    """Return multitask inference set for evaluation.
     
    :param model: trained tensorflow model satisfying abstract model interface
    :param dataset: tensorflow dataset wrapper
    :param sess: tensorflow session
    :return: multitask inference set object
    """
    inference_set = multitask_inference_set(total_tasks=TOTAL_DEEPSEA_TASKS)
    data_stream_op = _build_evaluation_datastream(dataset, sess)

    for _ in trange(dataset.number_of_examples, desc="multitask evaluation progress"):
        multitask_vp = _convert_example_to_multitask_validation_point(
                model=model, training_example=sess.run(data_stream_op), sess=sess)
        inference_set.add_multitask_validation_point(multitask_vp)

    return inference_set


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
                            attention_result_instance=attention_result(
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
