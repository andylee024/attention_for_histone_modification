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

    def add_multitask_validation_point(self, multitask_vp):
        """Unpack multitask_validation_point to individual tasks.
        
        :param multitask_vp: multitask validation point object
        """
        assert isinstance(multitask_vp, multitask_validation_point)
        assert len(multitask_vp) == len(self._tasks)
        
        singletask_vps = multitask_vp.single_task_validation_points
        for (task_id, singletask_vp) in enumerate(multitask_vp.single_task_validation_points):
            task = self.get_task_by_id(task_id)
            task.add_validation_point(singletask_vp)

    def get_task_by_id(self, task_id):
        """Retrieve inference set of a single task based on task_id
        
        :param task_id: int referring to task id
        :return: inference set
        """
        return self._tasks[task_id]
   
    @property
    def validation_points(self):
        """Return validation points aggregated across all tasks."""
        validation_points = []
        for task in self._tasks.itervalues():
            validation_points.extend(task.validation_points)
        return validation_points


class single_task_inference_set(object):
    """Class containing metrics for a single classification problem."""
    
    def __init__(self, task_id, task_name):
        """Initailize task scorer.
        
        :param task_id: Int. Identifies task in multitask settings. 
        :param task_name: Str. Description of prediction task.
        :param validation_points
        """
        self.task_id = task_id
        self.task_name = task_name
        self.validation_points = []

    def __len__(self):
        return len(self.validation_points)

    def add_validation_point(self, vp):
        """Add validation point to task.

        :param vp: validation point containing label and inference quantities
        """
        assert isinstance(vp, validation_point)
        self.validation_points.append(vp)

def _get_task_name_from_idx(idx):
    """Create name for task based on index."""
    return "task_{}".format(idx) 

