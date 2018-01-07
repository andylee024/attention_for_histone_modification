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
