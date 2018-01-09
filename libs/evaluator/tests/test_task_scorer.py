import unittest

from komorebi.libs.evaluator.metric_types import example_score
from komorebi.libs.evaluator.task_scorer import multitask_scorer, single_task_scorer

class test_multitask_scorer(unittest.TestCase):
    """Tests for multitask scorer."""

    def test_init_raises(self):
        """Test multitask raises when accessing uninitialized task."""
        mt_scorer = multitask_scorer(total_tasks=1)
        invalid_index = 1
        self.assertRaises(KeyError, mt_scorer.get_task_by_id, invalid_index)

    def test_add_example_score_raises(self):
        """Test multitask populates tasks."""

        # Expect error to raise when incorrect number of classifications and labels
        mt_scorer = multitask_scorer(total_tasks=1)
        classifications = [0, 0]
        labels = [1, 1, 1]
        self.assertRaises(AssertionError, mt_scorer.add_example_score, classifications, labels)

        # Expect error to raise when accessing uninitialized tasks
        mt_scorer = multitask_scorer(total_tasks=1)
        classifications = [0, 0, 0]
        labels = [1, 1, 1]
        self.assertRaises(AssertionError, mt_scorer.add_example_score, classifications, labels)

    def test_example_scores_property(self):
        """Test that examples scores are aggregated correctly."""
        classifications = [0, 0, 0]
        labels = [1, 1, 1]
        mt_scorer = multitask_scorer(total_tasks=3)
        mt_scorer.add_example_score(classifications, labels)

        expected_length = 3
        self.assertEqual(len(mt_scorer.example_scores), expected_length)
    
    def test_example_scores_stored_correctly(self):
        """Test that scores for each task is stored correctly."""

        classifications = [0, 0, 0]
        labels = [1, 1, 1]
        mt_scorer = multitask_scorer(total_tasks=3)
        mt_scorer.add_example_score(classifications, labels)

        expected_classification = 0
        expected_label = 1
        for index in range(len(classifications)):
            st_scorer = mt_scorer.get_task_by_id(index)
            es = st_scorer.example_scores[0]
            self.assertEqual(es.classification, expected_classification)
            self.assertEqual(es.label, expected_label)


class test_single_task_scorer(unittest.TestCase):
    """Tests for single task scorer."""

    def setUp(self):
        """Setup single_task_scorer task for tests."""
        self.single_task_scorer = single_task_scorer(task_id=0, task_name="dummy")
        
    def test_add_example_score(self):
        """Test annotation extraction for single sequence."""
        example_score_0 = example_score(classification=0, label=0)
        example_score_1 = example_score(classification=1, label=1)

        self.single_task_scorer.add_example_score(example_score_0)
        self.single_task_scorer.add_example_score(example_score_1)

        expected_example_scores = 2
        self.assertEqual(len(self.single_task_scorer.example_scores), expected_example_scores)

if __name__ == '__main__':
    unittest.main()
