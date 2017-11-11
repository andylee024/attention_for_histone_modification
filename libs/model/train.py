import abc

import collections
import numpy as np
import pickle
import tensorflow as tf

from attention_for_histone_modification.libs.model.attention_configuration import (
        AttentionConfiguration, LearningConfiguration)
from attention_for_histone_modification.libs.model.attention_model import AttentionModel
from attention_for_histone_modification.libs.preprocessing.utilities import load_pickle_object, partition_indices

def main():
    dataset = _get_dataset()

    shuffled_indices = _get_shuffled_indices(dataset.total_examples)
    index_batches, _ = partition_indices(shuffled_indices, 10)
    batch_size = len(index_batches[0])

    model = _get_model(batch_size)
    trainer = attention_trainer()

    print "training model"
    trainer.train_model(model, dataset)
    return




class abstract_trainer(object):
    """Interface for neural network training.

    The abstract_trainer takes a dataset and model and provides the logic for how
    to train the model.
    """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def train_model(model, dataset):
        """Train a model on a dataset."""

           
TrainingTensor = collections.namedtuple(typename="TrainingTensor", 
                                        field_names=['sequence_tensor', 'annotation_tensor', 'label_tensor'])

class attention_trainer(abstract_trainer):
    """Trainer for attention network."""

    def __init__(self):
        """Initialize trainer.
        
        :param trainer_config: configuration object specifying training parameters.
        """
        self._epochs = 1
        self._iterations = 10
        self._learning_rate = 0.01

        self._optimizer = self.get_optimizer(self._learning_rate)


    def train_model(self, model, dataset):
        """Train attention model."""
        
        # FIX THIS - HACK
        shuffled_indices = _get_shuffled_indices(dataset.total_examples)
        index_batches, _ = partition_indices(shuffled_indices, self._iterations)
        batch_size = len(index_batches[0])
        index_batches = index_batches[:-1]

        # setup operations
        model_inputs = model.get_model_inputs()
        loss_op = model.get_loss_op(model_inputs)
        train_op = self.get_train_op(self._optimizer, loss_op)

        # initialize variables
        init_op = tf.global_variables_initializer()

        # reuse variables
        tf.get_variable_scope().reuse_variables()

        # initialize session and start training
        with tf.Session() as sess:
            sess.run(init_op)
            for _ in xrange(self._epochs):

                # shuffle indices
                shuffled_indices = _get_shuffled_indices(dataset.total_examples)
                index_batches[:-1], _ = partition_indices(shuffled_indices, self._iterations)
                
                # compute iterations in epoch
                for idx, ib in enumerate(index_batches):
                    _, training_examples = zip(*dataset.get_training_examples(ib))
                    training_tensor = self.convert_to_training_tensor(training_examples)

                    feed_dict = {model_inputs['sequences']: training_tensor.sequence_tensor,
                                 model_inputs['features']: training_tensor.annotation_tensor,
                                 model_inputs['labels']: training_tensor.label_tensor}

                    _, loss_value = sess.run([train_op, loss_op], feed_dict)
                    print "the loss for iteration {} = {}".format(idx, loss_value)

    
    def convert_to_training_tensor(self, training_examples):
        """Convert training examples to training tensor for tf model.
        
        :param training_examples:
            List of attention training examples.
        :return:
            TrainingTensor object.
        """
        sequence_tensor = np.concatenate([np.expand_dims(te.sequence, axis=0) for te in training_examples], axis=0)
        label_tensor = np.concatenate([np.expand_dims(te.label, axis=0) for te in training_examples], axis=0)
        annotation_tensor = np.concatenate([np.reshape(te.annotation, (1, 1, te.annotation.size)) for te in training_examples], axis=0)
        return TrainingTensor(sequence_tensor=sequence_tensor,
                              annotation_tensor=annotation_tensor,
                              label_tensor=label_tensor)


    def get_optimizer(self, learning_rate):
        """Get optimizer for training."""
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


    def get_train_op(self, optimizer, loss_op):
        """Get tensorflow train op for attention model.
    
        :param loss_op: Tensorflow loss op
        :return: Tensorflow train op
        """
        with tf.name_scope('optimizer'):
            train_op = optimizer.minimize(loss_op)
            return train_op


def _get_dataset():
    """Get dataset used for training model.
    
    :param dataset_path: path to dataset pkl file.
    :return: AttentionDataset type object.
    """
    sharded_path = "/Users/andy/Projects/biology/research/attention_for_histone_modification/data/attention_validation_dataset/sharded_attention_dataset.pkl"
    return load_pickle_object(sharded_path)

def _get_model(batch_size):
    """Get model used for training.
    
    :return: AttentionModel
    """
    attention_config = AttentionConfiguration(batch_size=batch_size,
                                              sequence_length=1000,
                                              vocabulary_size=4,
                                              prediction_classes=919,
                                              number_of_annotations=1,
                                              annotation_size=925,
                                              hidden_state_dimension=112)
    learning_config = LearningConfiguration()
    return AttentionModel(attention_config=attention_config, learning_config=learning_config)
# ----------------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------------



def _get_shuffled_indices(number_examples):
    """Return shuffled indices for number of examples."""
    indices = np.arange(number_examples)
    np.random.shuffle(indices)
    return indices


if __name__ == "__main__":
    main()
