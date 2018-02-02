import abc

class abstract_tensorflow_model(object):
    """Abstract base class specific to tensorflow models."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def inference(self):
        """Return inference ops."""
        pass

    @abc.abstractproperty
    def inputs(self):
        """Return placeholder objects corresponding to tensorflow model inputs.

        Tensorflow models operate on tf.placeholder objects that represent inputs to
        the model. This needs to be exposed as a public interface so that those placeholders
        can be populated.
        """
        pass

    @abc.abstractproperty
    def outputs(self):
        """Return placeholder objects corresponding to tensorflow model outputs."""
        pass

    @abc.abstractproperty
    def prediction_signature(self):
        """Return prediction signature of model."""
        pass

    @abc.abstractmethod
    def _build_inference_graph(self, *args):
        """Build inference graph associated with model architecture."""
        pass

    @abc.abstractmethod
    def load_trained_model(self, trained_model_directory, sess):
        """Populate model ops based on trained model directory.

        :param trained_model_directory: directory containing tensorflow .pb trained model
        :param sess: tensorflow session
        """
        pass
