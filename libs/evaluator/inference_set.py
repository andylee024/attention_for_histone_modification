from komorebi.libs.evaluator.metric_types import multitask_validation_point, validation_point

class multitask_inference_set(object):
    """Class containing labels and inference quantities used for computing metrics for prediction tasks."""

    def __init__(self, total_tasks):
        """Initialize multitask inference set.

        :param total_tasks: total number of tasks in multitask problem
        """
        self._tasks = {
                idx: single_task_inference_set(task_id=idx, task_name=_get_task_name_from_idx(idx))
                for idx in range(total_tasks)}

    def add_multitask_validation_point(multitask_vp):
        """Unpack multitask_validation_point to individual tasks.
        
        :param multitask_vp: multitask validation point object
        """
        assert isinstance(multitask_vp, multitask_validation_point)
        assert len(multitask_vp) == len(self._tasks)
        for (task_id, singletask_vp) in enumerate(multitask_vp.single_task_validation_points):
            self._tasks[task_id].add_validation_point(singletask_vp)

    def get_task_by_id(self, task_id):
        """Retrieve inference set of a single task based on task_id
        
        :param task_id: int referring to task id
        :return: inference set
        """
        return self._tasks[task_id]
   
    @property
    def validation_points(self):
        """Return validation points of all tasks."""
        validation_points = []
        for task in self._tasks.itervalues():
            validation_points.extend(task.validation_points)
        return validation_points

    @property
    def classifications(self):
        """Return classifications of all tasks."""
        classifications = []
        for task in self_tasks.itervalues():
            classifications.extend(task.classifications)

    @property
    def labels(self):
        """Return labels of all tasks."""
        labels = []
        for task in self_tasks.itervalues():
            labels.extend(task.labels)

    @property
    def probability_predictions(self):
        """Return probability predictions of all tasks."""
        probability_predictions = []
        for task in self_tasks.itervalues():
            probability_predictions.extend(task.probability_predictions)



class single_task_inference_set(object):
    """Class containing metrics for a single classification problem."""
    
    def __init__(self, task_id, task_name, validation_points=[]):
        """Initailize task scorer.
        
        :param task_id: Int. Identifies task in multitask settings. 
        :param task_name: Str. Description of prediction task.
        """
        self.task_id = task_id
        self.task_name = task_name
        self.validation_points = validation_points

    def add_validation_point(self, vp):
        """Add validation point to task.

        :param vp: validation point containing label and inference quantities
        """
        assert isinstance(vp, validation_point)
        self.add_validation_points.append(vp)

    @property
    def classifications(self):
        return [vp.classification for vp in self.validation_points]

    @property
    def labels(self):
        return [vp.label for vp in self.validation_points]

    @property
    def probability_predictions(self):
        return [vp.probabiity_prediction for vp in self.validation_points]


def _get_task_name_from_idx(idx):
    """Create name for task based on index."""
    return "task_{}".format(idx) 