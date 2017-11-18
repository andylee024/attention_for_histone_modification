class ModelReturn(object):
    """Basic return type for machine learning models."""

    def __init__(self, predictions):
        self.predictions = predictions


class AttentionModelReturn(ModelReturn):
    """Return type for attention models."""

    def __init__(self, predictions, context_probabilities):
        """Constructor for attention return type.

        :param predictions: output from attention model
        :param context_probabilities: probabilities on annotation in attention model
        """
        self.predictions = predictions
        self.context_probabilities = context_probabilities

