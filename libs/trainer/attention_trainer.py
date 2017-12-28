import collections
import numpy as np
import tensorflow as tf

from komorebi.libs.trainer.abstract_tensorflow_trainer import AbstractTensorflowTrainer

class AttentionTrainer(AbstractTensorflowTrainer):
    """Trainer implementation for training attention models."""

    def _build_computational_graph(self, model, optimizer):
        """Construct a computational graph for training a model.
    
        :param model: tensorflow model to be trained
        :param optimizer: optimizer for gradient backpropogation
        :return: 
            2-tuple consisting the two dictionaries. The first dictionary contains tf.placeholders
            representing inputs to the graph. The second dictionary contains ops generated by the graph.
        """
        graph_inputs = {'features'  : model.inputs['features'],
                        'sequences' : model.inputs['sequences'],
                        'labels'    : model.outputs['labels']}
    
        model_return = model.predict(features=graph_inputs['features'], 
                                    sequences=graph_inputs['sequences'])
    
        loss_op = _get_loss_op(predictions=model_return.predictions, labels=graph_inputs['labels'])
        train_op = _get_train_op(loss_op=loss_op, optimizer=optimizer)
        summary_op = _get_summary_op(loss_op)
    
        ops = {'loss_op' : loss_op, 'train_op': train_op, 'summary_op': summary_op}
        return graph_inputs, ops

def _get_loss_op(predictions, labels):
    """Return loss for model.

    :param predictions: tensor ouput from model.
    :param labels: groundtruth labels.
    :return: loss
    """
    number_of_samples = tf.shape(labels)[0]
    total_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=labels))
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

