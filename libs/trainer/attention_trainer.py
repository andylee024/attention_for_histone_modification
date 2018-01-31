import tensorflow as tf

from komorebi.libs.trainer.abstract_tensorflow_trainer import AbstractTensorflowTrainer
from komorebi.libs.trainer.trainer_utils import infer_positive_upweight_parameter
from komorebi.libs.utilities.constants import SINGLE_PREDICTION

class AttentionTrainer(AbstractTensorflowTrainer):
    """Trainer implementation for training attention models."""

    def __init__(self, task_index, trainer_config, experiment_directory, checkpoint_directory, summary_directory):
        """Initialize trainer.

        :param task_index: the task index on which to train model
        :param config: trainer_config object.
        :param experiment_directory: directory corresponding to experiment
        :param checkpont_directory: directory for storing model checkpoints
        :param summary_directory: directory for storing tf summaries
        """
        super(AttentionTrainer, self).__init__(config=trainer_config,
                                               experiment_directory=experiment_directory,
                                               checkpoint_directory=checkpoint_directory,
                                               summary_directory=summary_directory)
        self.task_index = task_index

    def _build_computational_graph(self, dataset, model, optimizer):
        """Construct a computational graph for training a model.
    
        :param model: tensorflow model to be trained
        :param optimizer: optimizer for gradient backpropogation
        :return: 
            2-tuple consisting the two dictionaries. The first dictionary contains tf.placeholders
            representing inputs to the graph. The second dictionary contains ops generated by the graph.
        """
        positive_upweight = _infer_training_parameters(tf_dataset=dataset, task_id=task_id)

        graph_inputs = {'features'  : model.inputs['features'],
                        'sequence' : model.inputs['sequence'],
                        'labels'    : model.outputs['labels']}

        predictions, single_task_labels = _prepare_single_task_data(
                task_index=self.task_index, predictions=model.inference['logit'], labels=graph_inputs['labels'])

        loss_op = _get_loss_op(predictions=predictions, labels=single_task_labels, positive_upweight)
        train_op = _get_train_op(loss_op=loss_op, optimizer=optimizer)
        summary_op = _get_summary_op(loss_op)

        ops = {'loss_op' : loss_op, 'train_op': train_op, 'summary_op': summary_op}
        return graph_inputs, ops


    def _convert_training_examples(self, data, graph_inputs):
        """Convert training examples to graph inputs.
        
        :param data: tf examples parsed from dataset
        :param graph_inputs: dictionary mapping from string key to tf.placeholders
        :return: 
            feed_dict dictionary where keys are tf.placeholders and values are tensors.
            This dictionary is passed to computational graph.
        """
        return {graph_inputs['sequence']: data['sequence'],
                graph_inputs['features']: data['annotation'],
                graph_inputs['labels']: data['label']}


def _prepare_single_task_data(task_index, predictions, labels):
    """Extract relevant single task labels from the original dataset labels and make data shapes consistent.

    :param task_index: the task index corresponding to single task prediction
    :param predictions: tensor ouput from model.
    :param labels: groundtruth labels from original dataset
    :return: processed predictions, labels
    """
    single_task_labels = tf.reshape(labels[:, task_index], [-1, SINGLE_PREDICTION])
    predictions = tf.reshape(predictions, [-1, SINGLE_PREDICTION])
    return single_task_labels, predictions


def _get_loss_op(predictions, labels, positive_upweight):
    """Return loss for model.

    :param predictions: tensor ouput from model (currently logits)
    :param labels: groundtruth labels for single task prediction
    :param positive_upweight: upweight parameter for positive samples
    :return: loss
    """
    with tf.name_scope('loss'):
        number_of_samples = tf.shape(labels)[0]
        total_loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(
            targets=labels, logits=predictions, pos_weight=positive_upweight))
        return total_loss / tf.cast(number_of_samples, tf.float32)


def _get_train_op(optimizer, loss_op):
    """Perform gradient updates on model.

    :param loss_op: tensorflow loss op representing loss for which to compute gradients
    :return: tensorflow training op
    """
    with tf.name_scope('optimizer'):
        train_op = optimizer.minimize(loss_op)
        return train_op


def _get_summary_op(loss_op):
    """Summarize training statistics.

    :param loss_op: tensorflow loss op representing loss for which to compute gradients
    :return: tensorflow summary op
    """
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", loss_op)
        tf.summary.histogram("histogram_loss", loss_op)
        return tf.summary.merge_all()


def _infer_training_parameters(dataset, task_id):
    """Infer training parameters from dataset and task.

    Currently, the only training parameter we infer is the upweight parameter applied to positive samples.

    :param tf_dataset: dataset object
    :param task_id: task for which to compute statistics.
    :param sess: tensorflow session 
    :return: float. upweight parameter for positive samples
    """
    with tf.Session() as sess:
        return infer_positive_upweight_parameter(tf_dataset=dataset, task_id=task_id, sess=sess)

