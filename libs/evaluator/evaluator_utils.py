import collections
import tensorflow as tf

# Struct for holding inference ops of loaded model
inference_ops = collections.namedtuple(typename='inference_ops', 
                                       field_names=['sequence_placeholder', 'annotation_placeholder', 'prediction'])


def get_inference_ops(trained_model_config, sess):
    """Return inference ops for trained model.

    The tensorflow interface requires that trained models be loaded in sessions.
    For this reason, we return ops linked to a session for inference.
    
    :param trained_model_config: model config specifying trained model
    :param sess: tensorflow session
    """
    _load_inference_graph_into_session(trained_model_config, sess)
    graph = tf.get_default_graph()
    return inference_ops(
            sequence_placeholder=graph.get_tensor_by_name(trained_model_config.sequence_placeholder_op_name),
            annotation_placeholder=graph.get_tensor_by_name(trained_model_config.annotation_placeholder_op_name),
            prediction=graph.get_tensor_by_name(trained_model_config.prediction_op_name))


def _load_inference_graph_into_session(trained_model_config, sess):
    """Add inference graph and ops into session.
    
    The tensorflow API requires the inference graph to be loaded into the session.
    Note that this function mutates the tensorflow session by adding these ops.

    :param model_config: model config specifying trained model
    :param sess: tensorflow session
    """
    return tf.saved_model.loader.load(sess=sess, 
                                      tags=[tf.saved_model.tag_constants.SERVING],
                                      export_dir=trained_model_config.trained_model_directory)

