import tensorflow as tf

from attention_model import AttentionConfiguration, AttentionModel, LearningConfiguration
from mock_data import create_dummy_batch_data


def print_batch_data_dimension(attention_config):
    batch_data = create_dummy_batch_data(attention_config)
    print batch_data.sequence_tensor.shape
    print batch_data.annotation_tensor.shape
    print batch_data.label_tensor.shape


def main():
    # create attention model
    attention_config = AttentionConfiguration()
    learning_config = LearningConfiguration()
    attention_model = AttentionModel(attention_config=attention_config, learning_config=learning_config)

    # print batch data dimensions
    print_batch_data_dimension(attention_config)

    # training hyper-parameters
    learning_rate = 0.001
    number_epochs = 1
    number_iterations = 10

    # specify loss op
    model_inputs = attention_model.get_inputs()
    loss = attention_model.get_loss(model_inputs)

    # train op
    with tf.name_scope('optimizer'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

    # add initialization operations
    init_op = tf.global_variables_initializer()

    # reuse variables
    tf.get_variable_scope().reuse_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        for e in range(number_epochs):
            for i in range(number_iterations):
                batch_data = create_dummy_batch_data(attention_config)

                feed_dict = {model_inputs['sequences']: batch_data.sequence_tensor,
                             model_inputs['features']: batch_data.annotation_tensor,
                             model_inputs['labels']: batch_data.label_tensor}

                _, loss_value = sess.run([train_op, loss], feed_dict)
                print "the loss for iteration {} = {}".format(i, loss_value)


if __name__ == "__main__":
    main()
