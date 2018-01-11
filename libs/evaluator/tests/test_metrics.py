import unittest

from komorebi.libs.evaluator.inference_set import multitask_inference_set
from komorebi.libs.evaluator.metric_types import multitask_validation_point
from komorebi.libs.evaluator.metrics import compute_task_metrics

class test_metrics(unittest.TestCase):
    """Tests for multitask validation point object."""

    def setUp(self):
        """Common setup for unit tests."""
        classifications = [0, 0]
        labels = [0, 1]
        probabilities = [1.0, 1.0]

        self.multitask_vp = multitask_validation_point(
                classifications=classifications, probability_predictions=probabilities, labels=labels)

    def test_compute_metrics_for_multitask(self):
        """Test that compute metrics is evaluated correctly."""
        inference_set = multitask_inference_set(total_tasks=2)
        inference_set.add_multitask_validation_point(self.multitask_vp)

        task_metrics = compute_task_metrics(inference_set)

        expected_negative_examples = 1
        expected_positive_examples = 1
        expected_negative_accuracy = 1.0
        expected_positive_accuracy = 0.0
        expected_accuracy = 0.5

        self.assertEqual(task_metrics.negative_examples, expected_negative_examples)
        self.assertEqual(task_metrics.positive_examples, expected_positive_examples)
        self.assertEqual(task_metrics.negative_accuracy, expected_negative_accuracy)
        self.assertEqual(task_metrics.positive_accuracy, expected_positive_accuracy)
        self.assertEqual(task_metrics.accuracy, expected_accuracy)




    




if __name__ == '__main__':
    unittest.main()
