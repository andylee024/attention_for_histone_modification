
class example_score(object):
    """Struct containing scores for a single example."""

    def __init__(self, accuracy, precision=None, recall=None):
        """Initialize struct

        :param accuracy: accuarcy score
        :param precision: precision score
        :param recall: recall score
        """
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall

