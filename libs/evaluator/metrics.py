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

    return task_metrics(positive_examples=len(positive_inference_set),
                        negative_examples=len(negative_inference_set),
                        total_accuracy=_compute_accuracy(inference_set),
                        true_positive_rate=_compute_accuracy(positive_inference_set),
                        true_negative_rate=_compute_accuracy(negative_inference_set),
                        precision=_compute_precision(inference_set),
                        recall=_compute_recall(inference_set),
                        f1_score=_compute_f1_score(inference_set),
                        auroc=_compute_auroc(inference_set),
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


def _compute_auroc(inference_set):
    """Compute Area Under Receiver Operating Curve.

    :param inference_set: set of validation points to evaluate
    :return: AUROC between [0, 1]
    """
    probabilities, labels = zip(*[(vp.probability_prediction, vp.label) for vp in inference_set.validation_points])
    return sklearn.metrics.roc_auc_score(y_true=labels, y_score=probabilities)


def _compute_precision(inference_set):
    """Compute precision.
    
    Precision quantity captures accuracy among all positive predictions.

        ** precision = true_positive / (true_positive + false_positive) ** 

    :param inference_set: set of validation points to evaluate
    :return: precision score between [0, 1]
    """
    classifications, labels = zip(*[(vp.classification, vp.label) for vp in inference_set.validation_points])
    return sklearn.metrics.precision_score(y_true=labels, y_pred=classifications)


def _compute_recall(inference_set):
    """Compute recall.

    Recall captures the percentage of correct positive classifications among 
    all possible positive positive classifications in dataset.
    
    :param inference_set: set of validation points to evaluate
    :return: recall score between [0, 1]
    """
    classifications, labels = zip(*[(vp.classification, vp.label) for vp in inference_set.validation_points])
    return sklearn.metrics.recall_score(y_true=labels, y_pred=classifications)


def _compute_f1_score(inference_set):
    """Compute f1 score, a weighted average between precision and recall.

    :param inference_set: set of validation points to evaluate
    :return: f1 score
    """
    classifications, labels = zip(*[(vp.classification, vp.label) for vp in inference_set.validation_points])
    return sklearn.metrics.f1_score(y_true=labels, y_pred=classifications)


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

