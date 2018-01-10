"""Compute metrics for inference sets."""

import sklearn.metrics

from komorebi.libs.evaluator.inference_set import single_task_inference_set
from komorebi.libs.evaluator.metric_types import task_metrics 

def _create_inference_set(validation_points):
    """Create inference set based on validation points."""
    return single_task_inference_set(task_id="", task_name="", validation_points=validation_points)


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
            positive_accuracy=sklearn.metrics.accuracy_score(
                y_true=positive_inference_set.labels, y_pred=positive_inference_set.classifications, normalize=True),
            negative_accuracy=sklearn.metrics.accuracy_score(
                y_true=negative_inference_set.labels, y_pred=negative_inference_set.classifications, normalize=True),
            accuracy=sklearn.metrics.accuracy_score(
                y_true=inference_set.labels, y_pred=inference_set.classifications, normalize=True),
            cross_entropy_loss=sklearn.metrics.log_loss(
                y_true=inference_set.labels, y_pred=inference_set.probability_predictions)

