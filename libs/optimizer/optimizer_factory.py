import tensorflow as tf

from komorebi.libs.optimizer.optimizer_config import OptimizerConfiguration

def create_tf_optimizer(optimizer_config):
    """Create tensorflow optimizer from config.
    
    :param optimizer_config: OptimizerConfig object for creating optimizer.
    """
    assert isinstance(optimizer_config, OptimizerConfiguration)

    if optimizer_config.optimizer_type == "ada_grad":
        return tf.train.AdagradOptimizer(optimizer_config.learning_rate)

    elif optimizer_config.optimizer_type == "adam":
        return tf.train.AdamOptimizer(optimizer_config.learning_rate)

    elif optimizer_config.optimizer_type == "gradient_descent":
        return tf.train.GradientDescentOptimizer(optimizer_config.learning_rate)

    else:
        raise NotImplementedError("Unknown optimizer of type {}".format(optimizer_config.optimizer_type))

