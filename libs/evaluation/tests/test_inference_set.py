import unittest

from komorebi.libs.evaluator.inference_set import single_task_inference_set, multitask_inference_set
from komorebi.libs.evaluator.metric_types import multitask_validation_point, validation_point

class test_multitask_inference_set(unittest.TestCase):
    """Tests for multitask inference set."""

    def test_get_task_by_id_raises(self):
        """Test multitask raises when accessing uninitialized task."""
        inference_set = multitask_inference_set(total_tasks=1)
        invalid_id = 2
        self.assertRaises(KeyError, inference_set.get_task_by_id, invalid_id)

    def test_multitask_add_validation_point_raises(self):
        """Test that add multitask validation point raises for incorrect input types."""
        inference_set = multitask_inference_set(total_tasks=2)
        invalid_mt_vp = multitask_validation_point(classifications=[0, 0, 0],
                                           labels=[1, 1, 1],
                                           probability_predictions=[1.0, 1.0, 1.0])

        self.assertRaises(AssertionError, inference_set.add_multitask_validation_point, 1)
        self.assertRaises(AssertionError, inference_set.add_multitask_validation_point, "str")
        self.assertRaises(AssertionError, inference_set.add_multitask_validation_point, invalid_mt_vp)

    def test_multitask_validation_points_assigned_to_tasks_correctly(self):
        """Test that multitask validation points passed to single task correctly."""
        inference_set = multitask_inference_set(total_tasks=3)

        expected_validation_points = 3
        expected_classifications = [0, 0, 0]
        expected_labels = [1, 1, 1]
        expected_probability_predictions = [1.0, 1.0, 1.0]

        mt_vp = multitask_validation_point(classifications=expected_classifications,
                                           labels=expected_labels,
                                           probability_predictions=expected_probability_predictions)
        inference_set.add_multitask_validation_point(mt_vp)
       
        self.assertEqual(len(inference_set.validation_points), expected_validation_points)
        self.assertListEqual(inference_set.classifications, expected_classifications)
        self.assertListEqual(inference_set.probability_predictions, expected_probability_predictions)
        self.assertListEqual(inference_set.labels, expected_labels)


class test_single_task_inference_set(unittest.TestCase):
    """Tests for single task inference set."""

    def test_add_validation_point_raises(self):
        """Test that add validation point raises for incorrect input types."""
        inference_set = single_task_inference_set(task_id=0, task_name="dummy")
        self.assertRaises(AssertionError, inference_set.add_validation_point, 1)
        self.assertRaises(AssertionError, inference_set.add_validation_point, "str")
        

if __name__ == '__main__':
    unittest.main()
