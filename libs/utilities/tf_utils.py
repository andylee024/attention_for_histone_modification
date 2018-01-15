import tensorflow as tf

def load_inference_graph_into_session(trained_model_directory, sess):
    """Add inference graph and ops into session.
    
    The tensorflow API requires the inference graph to be loaded into the session.

    :param trained_model_directory: directory containing trained model .pb file
    :param sess: tensorflow session
    :return: tensorflow graph containing inference ops
    """
    tf.saved_model.loader.load(sess=sess, 
                               tags=[tf.saved_model.tag_constants.SERVING],
                               export_dir=trained_model_directory)
    return tf.get_default_graph()


def load_checkpoint_into_session(model_checkpoint, sess):
    """Load saved model graph into session.

    :param model_checkpoint: tensorflow model_checkpoint file
    :param sess: tensorflow session
    :return: tensorflow graph containing recovered ops
    """
    saver = tf.train.import_meta_graph(_convert_checkpoint_to_meta_graph_file(model_checkpoint))
    saver.restore(sess, model_checkpoint)
    return tf.get_default_graph()


def _convert_checkpoint_to_meta_graph_file(model_checkpoint):
    """Convert model checkpoint to corresponding meta graph file.
    
    :param model_checkpoint: tensorflow model_checkpoint file
    :return: metagraph file
    """
    return model_checkpoint + ".meta"
