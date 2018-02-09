from komorebi.libs.evaluation.model_scoring_utils import (
        generate_multitask_inference_set, generate_single_task_inference_set)
from komorebi.libs.evaluation.metrics import compute_task_metrics


def score_multitask_model(model, dataset, sess):
    """Evaluate multitask model on dataset.
    
    :param model: trained tensorflow model satisfying abstract model interface
    :param dataset: tf_dataset_wrapper object of dataset on which to evaluate model
    :param sess: tensorflow session
    :return: 2-tuple (inference_set, task_metrics)
    """
    inference_set = generate_multitask_inference_set(model=model, 
                                                     dataset=dataset, 
                                                     sess=sess)
    return compute_task_metrics(inference_set), inference_set


def score_single_task_model(task_id, model, dataset, sess):
    """Evaluate single task model on dataset.

    :param task_id: task_id that identifies predictive task
    :param model: trained tensorflow model satisfying abstract model interface
    :param dataset: tf_dataset_wrapper object of dataset on which to evaluate model
    :param sess: tensorflow session
    :return 2-tuple (inference_set, task_metrics)
    """
    inference_set = generate_single_task_inference_set(task_id=task_id,
                                                       model=model,
                                                       dataset=dataset,
                                                       sess=sess)
    return compute_task_metrics(inference_set), inference_set


