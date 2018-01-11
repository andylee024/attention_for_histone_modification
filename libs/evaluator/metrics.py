"""Compute metrics for inference sets."""

import sklearn.metrics

from komorebi.libs.evaluator.inference_set import single_task_inference_set
from komorebi.libs.evaluator.metric_types import task_metrics, validation_point


def compute_task_metrics(inference_set):
    """Compute task metrics for inference set.

    :param inference_set: single task or multitask inference set
    :return: task_metrics object
    """
    positive_inference_set = _create_inference_set([vp for vp in inference_set.validation_points if vp.label==1])
    negative_inference_set = _create_inference_set([vp for vp in inference_set.validation_points if vp.label==0])

    return task_metrics(
            positive_examples=len(positive_inference_set),
            negative_examples=len(negative_inference_set),
            positive_accuracy=_compute_accuracy(positive_inference_set),
            negative_accuracy=_compute_accuracy(negative_inference_set),
            accuracy=_compute_accuracy(inference_set),
            cross_entropy=_compute_cross_entropy(inference_set))


def _create_inference_set(validation_points):
    """Create inference set based on validation points.
    
    :param validation_points: list of validation point objects
    :return: inference set initialized with supplied validation points
    """
    assert all((isinstance(vp, validation_point) for vp in validation_points))
    inference_set = single_task_inference_set(task_id="", task_name="")
    inference_set.validation_points = validation_points
    return inference_set


def _compute_accuracy(inference_set):
    """Compute accuracy of inference set.
    
    :param inference_set: set of validation points to evaluate
    :return: accuracy between [0, 1]
    """
    classifications, labels = zip(*[(vp.classification, vp.label) for vp in inference_set.validation_points])
    return sklearn.metrics.accuracy_score(y_true=labels, y_pred=classifications, normalize=True)


def _compute_cross_entropy(inference_set):
    """Compute cross entropy loss of inference set.
    
    :param inference_set: set of validation points to evaluate
    :return: cross-entropy loss
    """
    probabilities, labels = zip(*[(vp.probability_prediction, vp.label) for vp in inference_set.validation_points])
    return sklearn.metrics.log_loss(y_true=labels, y_pred=probabilities)

