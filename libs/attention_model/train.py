import pickle
import tensorflow as tf

from attention_model import AttentionConfiguration, AttentionModel, LearningConfiguration
from mock_data import create_dummy_batch_data


def main():

    #training_examples = load_attention_training_examples()
    
    # create attention model configurations
    attention_config = AttentionConfiguration(batch_size=100,
                                              sequence_length=1000,
                                              vocabulary_size=4,
                                              prediction_classes=919,
                                              number_of_annotations=1,
                                              annotation_size=925,
                                              hidden_state_dimension=112)
    learning_config = LearningConfiguration()
    
    # print batch data dimensions
    print_batch_data_dimension(attention_config)

    # specify training parameters
    number_epochs = 1
    number_iterations = 10

    # create attention model
    attention_model = AttentionModel(
        attention_config=attention_config,
        learning_config=learning_config)

    # specify tensorflow ops
    model_inputs = attention_model.get_model_inputs()
    loss_op = attention_model.get_loss_op(model_inputs)
    train_op = get_train_op(loss_op)

    # initialize variables
    init_op = tf.global_variables_initializer()

    # reuse variables
    tf.get_variable_scope().reuse_variables()

    # initialize session and start training
    with tf.Session() as sess:
        sess.run(init_op)
        for e in range(number_epochs):
            for i in range(number_iterations):
                batch_data = create_dummy_batch_data(attention_config)

                feed_dict = {model_inputs['sequences']: batch_data.sequence_tensor,
                             model_inputs['features']: batch_data.annotation_tensor,
                             model_inputs['labels']: batch_data.label_tensor}

                _, loss_value = sess.run([train_op, loss_op], feed_dict)
                print "the loss for iteration {} = {}".format(i, loss_value)


# ----------------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------------

def load_attention_training_examples():
    """Load training examples for attention model."""
    dataset_path = "/Users/andy/Projects/bio_startup/research/attention_for_histone_modification/data/annotated_validation_dataset.pkl"
    with open(dataset_path, 'r') as f:
        dataset = pickle.load(f)

        print "dataset loaded..."
        print "sequence shape: {}".format(dataset.training_examples[0].sequence.shape)
        print "label shape: {}".format(dataset.training_examples[0].label.shape)
        print "annotation shape: {}".format(dataset.training_examples[0].annotation.shape)

        return dataset.training_examples


def print_batch_data_dimension(attention_config):
    """Print batch data dimensions for debugging purposes."""
    batch_data = create_dummy_batch_data(attention_config)
    print "sequence tensor shape {}".format(batch_data.sequence_tensor.shape)
    print "annotation tensor shape {}".format(batch_data.annotation_tensor.shape)
    print "label tensor shape {}".format(batch_data.label_tensor.shape)


def get_train_op(loss_op):
    """Get tensorflow train op for attention model.

    :param loss_op: Tensorflow loss op
    :return: Tensorflow train op
    """
    learning_rate = 0.001
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        return train_op


if __name__ == "__main__":
    main()
