import unittest

from komorebi.libs.evaluator.metric_types import multitask_validation_point, validation_point

class test_mulittask_validation_point(unittest.TestCase):
    """Tests for multitask validation point object."""

    def _verify_same_validation_point(self, target_vp, expected_vp):
        """Verify that the two supplied validation points are the same."""
        self.assertEqual(target_vp.classification, expected_vp.classification)
        self.assertEqual(target_vp.probability_prediction, expected_vp.probability_prediction)
        self.assertEqual(target_vp.label, expected_vp.label)

    def test_constructor_raises(self):
        """Test constructor raises assertion error when input argument lengths do not match."""
        classifications = [0, 0]
        labels = [1, 1, 1]
        probability_predictions = [0.0]
        self.assertRaises(AssertionError, multitask_validation_point, classifications, probability_predictions, labels)

    def test_single_task_validation_points(self):
        """Test that single task validation points are retrieved correctly."""
        classifications = [0, 0, 0]
        labels = [1, 1, 1]
        probability_predictions = [1.0, 1.0, 1.0]

        multitask_vp = multitask_validation_point(classifications=classifications,
                                                  probability_predictions=probability_predictions,
                                                  labels=labels)
        
        expected_vp = validation_point(classification=0, label=1, probability_prediction=1.0)
        for single_task_vp in multitask_vp.single_task_validation_points:
            self._verify_same_validation_point(single_task_vp, expected_vp)

    def test_multitask_validation_properties(self):
        """Test that properties are defined correctly for multitask validation points."""
        classifications = [0, 0, 0]
        labels = [1, 1, 1]
        probability_predictions = [1.0, 1.0, 1.0]

        multitask_vp = multitask_validation_point(classifications=classifications,
                                                  probability_predictions=probability_predictions,
                                                  labels=labels)

        self.assertListEqual(multitask_vp.classifications, classifications)
        self.assertListEqual(multitask_vp.labels, labels)
        self.assertListEqual(multitask_vp.probability_predictions, probability_predictions)


if __name__ == '__main__':
    unittest.main()
