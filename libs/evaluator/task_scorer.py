from komorebi.libs.evaluator.metric_types import example_score

class multitask_scorer(object):
    """Class containing metrics for multitask classification problem."""

    def __init__(self, total_tasks):
        """Initialize multitask scorer

        :param total_tasks: total number of tasks in multitask problem
        """
        self._tasks = {
                idx: single_task_scorer(task_id=idx, task_name="task_{}".format(idx))
                for idx in range(total_tasks)}

    def add_example_score(self, classifications, labels):
        """Add example score for single multitask example.

        Note this function assumes that the classifcations and labels are matching based off indices.
        """
        assert len(self._tasks) == len(classifications)
        assert len(self._tasks) == len(labels)
        for (idx, (classification, label)) in enumerate(zip(classifications, labels)):
            self.get_task_by_id(idx).add_example_score(example_score(classification=classification, label=label))

    def get_task_by_id(self, task_id):
        """Retrieve single task by id."""
        return self._tasks[task_id]

    @property
    def example_scores(self):
        scores = []
        for task in self._tasks.itervalues():
            scores.extend(task.example_scores)
        return scores


class single_task_scorer(object):
    """Class containing metrics for a single classification problem."""
    
    def __init__(self, task_id, task_name):
        """Initailize task scorer.
        
        :param task_id: Int. Identifies task in multitask settings. 
        :param task_name: Str. Description of prediction task.
        """
        self.task_id = task_id
        self.task_name = task_name

        self.example_scores = []

    def add_example_score(self, validation_example_score):
        """Add score for an example corresponding to prediction task.

        :param validation_example_score: struct containing classification and label for task
        """
        assert isinstance(validation_example_score, example_score)
        self.example_scores.append(validation_example_score)

